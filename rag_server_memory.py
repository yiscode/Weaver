import os
import json
import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import logging

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from pydantic import BaseModel

import weaviate
from weaviate.classes.query import Filter

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# === Logging 設定 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("weaver_rag")


# === 環境與 LLM 設定 ===
LLM_URL = os.getenv("LLM_URL", "http://localhost:1337/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-2-9b-it-abliterated-IQ4_XS")
DEBUG_TOKEN = "weaver_admin_2025"

# Weaviate client（同步版）
client = weaviate.connect_to_local(
    host=os.getenv("WEAVIATE_HOST", "localhost"),
    port=9080,
    grpc_port=50051,
    skip_init_checks=True,
)

log.info(f"🚀 Weaver Engine 啟動：多世界觀 RAG (Model: {LLM_MODEL})")

text_embedder = SentenceTransformer("BAAI/bge-m3")

llm = ChatOpenAI(
    base_url=LLM_URL,
    api_key="my-secret-key",
    model=LLM_MODEL,
    temperature=0.75,
    max_tokens=650,
    top_p=0.9,
    frequency_penalty=1.4,
    presence_penalty=1.1,
)

app = FastAPI(title="Multi-Worldview RAG Server")


# === 工具函式 ===

def to_traditional_zh(text: str) -> str:
    try:
        from opencc import OpenCC
        return OpenCC("s2twp").convert(text)
    except Exception:
        return text


def robust_json_decode(s: str) -> Optional[dict]:
    """盡量從 LLM 回傳文字中抽出 JSON。"""
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if not match:
        return None
    clean_s = match.group(0).replace("'", '"')
    clean_s = re.sub(r"\n", " ", clean_s)
    clean_s = re.sub(r",\s*\}", "}", clean_s)
    try:
        return json.loads(clean_s)
    except Exception:
        # 最後保底：只抓 summary
        sum_m = re.search(r'"summary":\s*"([^"]*)"', clean_s)
        return {
            "summary": sum_m.group(1) if sum_m else "劇情推進中",
            "timeline_append": [],
            "characters_update": [],
            "flags_update": []
        }


# === 世界觀管理 ===

class WorldviewManager:
    def __init__(self, base_dir: str = "worldviews"):
        self.base_dir = base_dir
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_worldviews(self) -> List[str]:
        if not os.path.isdir(self.base_dir):
            return []
        return [
            f[:-5] for f in os.listdir(self.base_dir)
            if f.endswith(".json")
        ]

    def load(self, name: str) -> Dict[str, Any]:
        if name in self._cache:
            return self._cache[name]
        path = os.path.join(self.base_dir, f"{name}.json")
        if not os.path.exists(path):
            raise ValueError(f"Unknown worldview: {name}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self._cache[name] = cfg
        log.info(f"[Worldview] Loaded config name={name}")
        return cfg


worldview_manager = WorldviewManager()


def detect_style_from_lore(worldview_cfg: dict, first_lore_text: str) -> str:
    default_style = worldview_cfg.get("style", {}).get("default_style", "泛用冒險")
    rules = worldview_cfg.get("style", {}).get("detection_rules", [])
    for rule in rules:
        if any(k in first_lore_text for k in rule.get("keywords", [])):
            return rule.get("style", default_style)
    return default_style


# === 身分與存檔管理 ===

def _generate_binding_key(session_id: str, player_id: str) -> str:
    return hashlib.md5(f"{session_id}_{player_id}".encode()).hexdigest()


def _get_pc_name(session_id: str, binding_key: str) -> Optional[str]:
    reg = client.collections.get("PCRegistry")
    f = (
        Filter.by_property("session_id").equal(session_id)
        & Filter.by_property("player_key").equal(binding_key)
    )
    objs = reg.query.fetch_objects(limit=1, filters=f).objects
    if objs:
        log.info(f"[PC] Found pc_name={objs[0].properties.get('pc_name')} for session={session_id}")
    else:
        log.info(f"[PC] No pc_name found for session={session_id}")
    return objs[0].properties.get("pc_name") if objs else None


def _upsert_pc_name(session_id: str, binding_key: str, pc_name: str):
    reg = client.collections.get("PCRegistry")
    f = Filter.by_property("player_key").equal(binding_key)
    deleted = reg.data.delete_many(where=f)
    log.info(f"[PC] Delete old PCRegistry rows count={getattr(deleted, 'matches', None)} session={session_id}")
    reg.data.insert(
        properties={
            "session_id": session_id,
            "player_key": binding_key,
            "pc_name": pc_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    log.info(f"[PC] Upsert pc_name={pc_name} session={session_id}")


# === 世界狀態管理（結構化） ===

def _get_state_obj(session_id: str) -> Optional[Any]:
    ws_coll = client.collections.get("WorldState")
    log.info(f"[WorldState] Fetch for session_id={session_id}")

    # 先抓所有，找最大 version
    all_objs = ws_coll.query.fetch_objects(
        limit=10,  # 最多10筆就好
        filters=Filter.by_property("session_id").equal(session_id),
    ).objects

    if not all_objs:
        log.info(f"[WorldState] No state for {session_id}")
        return None

    # 找最新 version
    latest_obj = max(all_objs, key=lambda obj: int(obj.properties.get('version', 0)))
    log.info(f"[WorldState] Latest v={latest_obj.properties.get('version')} ts={latest_obj.properties.get('timestamp')}")
    return latest_obj



def _empty_state() -> dict:
    return {
        "summary": "故事剛開始",
        "timeline": [],
        "characters": [],
        "flags": {}
    }


def _merge_state(existing: dict, delta: dict) -> dict:
    # 1. summary：直接覆蓋
    if "summary" in delta and delta["summary"]:
        existing["summary"] = delta["summary"]

    # 2. timeline：APPEND 新事件，保留歷史
    existing_tl = existing.get("timeline", [])
    new_events = delta.get("timeline_append", [])
    if new_events:  # ✅ 只在有新事件時才 append
        existing_tl.extend(new_events)
        existing["timeline"] = existing_tl[-50:]  # 保留最近50筆

    # 3. flags：合併
    flags = existing.get("flags", {})
    for f in delta.get("flags_update", []):
        flags[f.get("key")] = f.get("value")
    existing["flags"] = flags

    # 4. characters：以 name 為 key 合併
    existing_chars = {c.get("name"): c for c in existing.get("characters", [])}
    for new_char in delta.get("characters_update", []):
        name = new_char.get("name")
        if name:
            existing_chars[name] = {**existing_chars.get(name, {}), **new_char}
    existing["characters"] = list(existing_chars.values())

    return existing



def _summarize_state_for_prompt(state: dict) -> str:
    """可選：壓縮成 prompt 用的摘要文字。"""
    summary = state.get("summary", "故事剛開始")
    recent_events = " / ".join(state.get("timeline", [])[-3:])
    if recent_events:
        return f"{summary}（最近發生：{recent_events}）"
    return summary


def background_update_logic(
    session_id: str,
    new_content: str,
    pc_name: str,
    user_query: str,
    binding_key: str,
):
    time.sleep(2)
    log.info(f"[Memory Sync 🔍] 1️⃣ START session={session_id}")

    try:
        # Step 1: Get collection
        ws_coll = client.collections.get("WorldState")
        log.info(f"[Memory Sync 🔍] 2️⃣ Got collection OK")

        # Step 2: Get existing state
        ws_obj = _get_state_obj(session_id)
        if ws_obj:
            existing_state = json.loads(ws_obj.properties.get("state_json", "{}"))
            current_version = int(ws_obj.properties.get("version", 0))
            log.info(f"[Memory Sync 🔍] 3️⃣ Existing v={current_version}")
        else:
            existing_state = _empty_state()
            current_version = 0
            log.info(f"[Memory Sync 🔍] 3️⃣ No state, v=0")

        # Step 4: Build prompt
        log.info(f"[Memory Sync 🔍] 4️⃣ Building prompt...")
        update_prompt = f"""
自動從敘事中識別所有出現的人物，包括主角、NPC。

格式：
{{
  "summary": "單句摘要",
  "timeline_append": ["事件1"],
  "characters_update": [{{"name": "巴耶爾", "state": "狀態描述"}}],
  "flags_update": []
}}

內容：{new_content[:3000]}
"""
        log.info(f"[Memory Sync 🔍] 5️⃣ Prompt ready, len={len(update_prompt)}")

        # Step 5: LLM call
        log.info(f"[Memory Sync 🔍] 6️⃣ Calling LLM...")
        messages = [
            SystemMessage(content='嚴格只回 JSON，包含所有 4 個 keys'),
            HumanMessage(content=update_prompt),
        ]
        log.info(f"[Memory Sync 🔍] 7️⃣ Messages OK, calling invoke...")

        raw_res = llm.invoke(messages).content
        log.info(f"[Memory Sync 🔍] 8️⃣ LLM SUCCESS: {raw_res[:150]}...")

        # Step 6: Parse
        delta = robust_json_decode(raw_res) or {}
        log.info(f"[Memory Sync 🔍] 9️⃣ Delta keys={list(delta.keys())}")

        # Step 7: Merge
        merged_state = _merge_state(existing_state, delta)
        new_version = current_version + 1
        log.info(f"[Memory Sync 🔍] 🔟 Merged v={new_version}, chars={len(merged_state.get('characters', []))}")

        # Step 8: Insert
        log.info(f"[Memory Sync 🔍] 1️⃣1️⃣ Inserting...")
        result = ws_coll.data.insert(
            properties={
                "session_id": session_id,
                "state_json": json.dumps(merged_state, ensure_ascii=False),
                "summary": to_traditional_zh(merged_state.get("summary", "劇情更新")),
                "version": new_version,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        log.info(f"[Memory Sync ✅] 1️⃣2️⃣ INSERT SUCCESS! id={getattr(result, 'object_id', 'N/A')} v={new_version}")

    except Exception as e:
        log.error(f"🔴 [Memory Sync FAIL] Step failed: {e}", exc_info=True)
        import traceback
        log.error(f"🔴 Full traceback:\n{traceback.format_exc()}")


# === Prompt 組裝 ===

def render_system_prompt(
    pc_name: str,
    detected_style: str,
    worldview_cfg: dict,
    causal_logic: str,
    lore_text: str,
    state: dict,
) -> str:
    lang_cfg = worldview_cfg.get("language", {})
    resp_cfg = worldview_cfg.get("response", {})

    forbid_second_person = lang_cfg.get("forbid_second_person", True)
    forbid_first_person = lang_cfg.get("forbid_first_person", True)
    min_chars = resp_cfg.get("min_chars", 250)
    max_chars = resp_cfg.get("max_chars", 450)
    choices_count = resp_cfg.get("choices_count", 3)

    lang_rules = []
    if forbid_second_person:
        lang_rules.append("嚴禁使用「你」稱呼玩家。")
    if forbid_first_person:
        lang_rules.append("嚴禁使用「我」做為敘事主體。")

    lang_rule_str = "\n".join(f"- {r}" for r in lang_rules) or "- 使用自然的繁體中文敘事。"

    state_for_prompt = json.dumps(state, ensure_ascii=False)[:2000]
    ws_summary = _summarize_state_for_prompt(state)

    system_prompt = f"""
# 角色設定：【{pc_name}】
# 當前風格：【{detected_style}】
# 世界觀：【{worldview_cfg.get("name", "未命名")}】

# 當前世界狀態（結構化摘要）：
{state_for_prompt}

# 敘事摘要：
{ws_summary}

# 語言與敘事規則：
{lang_rule_str}
- 字數控制在約 {min_chars}-{max_chars} 字。
- 著重心理描寫、互動與環境細節。

# 回應結構：
1. 直接從「玩家行動造成的後果」開始描述，不要重覆轉述玩家剛才說的話。
2. 根據因果關係描述 NPC 或環境的反應，並加入新的資訊或線索。
3. 最後提供 {choices_count} 個具決策意義的下一步行動選項（使用編號列出）。

# 因果寫作核心：
{causal_logic}

# 參考 Lore（如有）：
{lore_text if lore_text else "無特定 Lore。請依世界觀與當前狀態產出合乎邏輯的新情節。"}
""".strip()

    return system_prompt


# === API 模型 ===

class WorldQuery(BaseModel):
    query: str
    user_name: str = "Player"
    player_id: Optional[str] = None
    session_id: str = "default"
    worldview: str = "generic_adventure"


class WorldResponse(BaseModel):
    content: str
    pc_name: str
    worldview: str


# === API 入口 ===

@app.get("/worldviews")
def list_worldviews():
    wvs = worldview_manager.list_worldviews()
    log.info(f"[Worldviews] list={wvs}")
    return {"worldviews": wvs}


@app.get("/state")
def get_state(session_id: str = Query(...)):
    ws_obj = _get_state_obj(session_id)
    if not ws_obj:
        log.info(f"[State API] No state for session_id={session_id}, return empty")
        return {"session_id": session_id, "state": _empty_state()}
    state = json.loads(ws_obj.properties.get("state_json", "{}"))
    log.info(
        f"[State API] Return state for session_id={session_id} "
        f"summary={state.get('summary')}"
    )
    return {"session_id": session_id, "state": state}


@app.post("/world", response_model=WorldResponse)
async def chat_world(q: WorldQuery, background_tasks: BackgroundTasks):
    # 0. 載入世界觀設定
    log.info(
        f"[World] Request: session_id={q.session_id} user={q.user_name} "
        f"worldview={q.worldview} query={q.query}"
    )
    try:
        worldview_cfg = worldview_manager.load(q.worldview)
        log.info(f"[World] Worldview loaded: {worldview_cfg.get('name','?')}")
    except ValueError as e:
        log.error(f"[World] Worldview load error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    ws_coll = client.collections.get("WorldState")

    # 1. 身分與 PC 名稱
    binding_key = _generate_binding_key(q.session_id, q.player_id or q.user_name)
    bracket_match = re.search(r"[\[【](.*?)[\]】]", q.query)
    if bracket_match:
        pc_name = bracket_match.group(1).strip()
        _upsert_pc_name(q.session_id, binding_key, pc_name)
    else:
        pc_name = _get_pc_name(q.session_id, binding_key) or "主角"

    log.info(f"[World] PC resolved: pc_name={pc_name} binding_key={binding_key}")

    # 2. 讀取世界狀態
    ws_obj = _get_state_obj(q.session_id)
    if ws_obj:
        state = json.loads(ws_obj.properties.get("state_json", "{}"))
    else:
        state = _empty_state()

    ws_summary = _summarize_state_for_prompt(state)
    log.info(
        f"[World] Current summary: {ws_summary} | "
        f"timeline_len={len(state.get('timeline', []))} "
        f"flags={list(state.get('flags', {}).keys())}"
    )

    # 3. RAG：依世界觀設定檢索 Lore
    lore_cfg = worldview_cfg.get("lore", {})
    lore_collection_name = lore_cfg.get("collection", "WorldLoreV2")
    lore_limit = int(lore_cfg.get("max_results", 3))

    lore_text = ""
    detected_style = worldview_cfg.get("style", {}).get("default_style", "泛用冒險")

    if client.collections.exists(lore_collection_name):
        log.info(f"[World] Using lore collection={lore_collection_name}")
        lore_coll = client.collections.get(lore_collection_name)

        recent_events = " / ".join(state.get("timeline", [])[-3:])
        flags_str = ", ".join(
            [k for k, v in state.get("flags", {}).items() if v]
        )

        search_str = f"{ws_summary} {recent_events} {flags_str} {q.query}"
        log.info(f"[World] RAG search_str={search_str[:200]}")
        vector = text_embedder.encode(search_str)
        log.info(f"[World] RAG vector shape={getattr(vector, 'shape', None)}")

        lore_res = lore_coll.query.near_vector(
            near_vector=vector,
            limit=lore_limit
        ).objects
        log.info(f"[World] RAG hits={len(lore_res)}")

        if lore_res:
            lore_text = "\n".join(
                [f"- 設定內容：{r.properties.get('text_zh', '')}" for r in lore_res]
            )
            first_lore = lore_res[0].properties.get("text_zh", "")
            detected_style = detect_style_from_lore(worldview_cfg, first_lore)
            log.info(f"[World] Detected style={detected_style}")
    else:
        log.warning(f"[World] Lore collection not exists: {lore_collection_name}")

    # 4. 因果指令（保持你的核心思想）
    causal_logic = f"""
玩家目前的動作或問題：『{q.query}』。
目前所處的情境摘要：『{ws_summary}』。

寫作方針：
- 禁止重覆描述玩家剛才的台詞或行動。
- 著重「行動後產生的結果」、NPC 的反應、環境的變化。
- 根據世界觀與 Lore，提供一個玩家尚未掌握的新線索。
- 加入具體感官描寫（氣味、聲音、觸感、光線變化等）。
""".strip()

    # 5. 組裝 System Prompt
    log.info("[World] Building system prompt...")
    system_prompt = render_system_prompt(
        pc_name=pc_name,
        detected_style=detected_style,
        worldview_cfg=worldview_cfg,
        causal_logic=causal_logic,
        lore_text=lore_text,
        state=state,
    )

    # 6. 生成內容
    try:
        log.info("[World] Calling LLM...")
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"玩家行動：{q.query}"),
            ]
        ).content
        log.info(f"[World] LLM response head={response[:120].replace(chr(10),' ')}")
    except Exception as e:
        log.error(f"[World] LLM error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="LLM invocation failed")

    response = to_traditional_zh(response)

    # 7. 背景更新世界狀態
    background_tasks.add_task(
        background_update_logic,
        q.session_id,
        response,
        pc_name,
        q.query,
        binding_key,
    )
    log.info(f"[World] Response sent. pc_name={pc_name}")

    return WorldResponse(content=response, pc_name=pc_name, worldview=q.worldview)


# === 維護接口 ===

@app.post("/reset")
def reset_session(session_id: str = Query(...)):
    log.info(f"[Reset] Reset session_id={session_id}")
    for col in ["WorldState", "SessionMemory", "PCRegistry"]:
        if client.collections.exists(col):
            client.collections.get(col).data.delete_many(
                where=Filter.by_property("session_id").equal(session_id)
            )
            log.info(f"[Reset] Cleared collection={col} session_id={session_id}")
        else:
            log.info(f"[Reset] Collection not exists: {col}")
    return {"msg": f"Session {session_id} reset"}


if __name__ == "__main__":
    import uvicorn

    log.info("Starting uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=9527)
