# rag_server_memory_v2.py
import os, json
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, Header
from pydantic import BaseModel

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === Settings ===
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "bge-reranker-base")
LLM_URL = os.getenv("LLM_URL", "http://localhost:1337/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Llama-3_1-8B-Instruct-IQ4_XS")

# Optional debug token header: X-Debug-Token
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")

# Weaviate
client = weaviate.connect_to_local(
    host=os.getenv("WEAVIATE_HOST", "localhost"),
    port=int(os.getenv("WEAVIATE_PORT", "9080")),
    grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
    skip_init_checks=True,
)

print("Loading Models (this may take a while)...")
text_embedder = SentenceTransformer(EMBED_MODEL)
try:
    reranker = CrossEncoder(RERANK_MODEL)
except Exception:
    reranker = None
    print("Warning: Reranker not found, skipping.")

llm = ChatOpenAI(
    base_url=LLM_URL,
    api_key=os.getenv("LLM_API_KEY", "my-secret-key"),
    model=LLM_MODEL,
    temperature=0.88,
    max_tokens=3000,
    # 直接寫在這裡，不要包在 model_kwargs 裡
    top_p=0.98,
    frequency_penalty=0.8,
    presence_penalty=0.6,
)

app = FastAPI()

# --- utils ---

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return str(v)

# OpenCC (Traditional Chinese)
try:
    from opencc import OpenCC
    _cc = OpenCC('s2twp')
except Exception:
    _cc = None

def to_traditional_zh(text: str) -> str:
    if not text:
        return text
    return _cc.convert(text) if _cc else text

import re

def light_dedup(text: str) -> str:
    if not text:
        return text
    # 針對句子進行去重
    parts = re.split(r"(?<=[。！？])", text)
    seen = set()
    out = []
    for p in parts:
        p_strip = p.strip()
        if not p_strip: continue
        # 如果句子核心內容（前10個字）已經出現過，就直接捨棄
        core = p_strip[:10]
        if core in seen:
            continue
        seen.add(core)
        out.append(p_strip)
    return "".join(out)


def require_debug_token(x_debug_token: Optional[str]):
    if DEBUG_TOKEN:
        if not x_debug_token or x_debug_token != DEBUG_TOKEN:
            raise HTTPException(status_code=401, detail="Unauthorized")


# --- schema ---

def ensure_schemas():
    if not client.collections.exists("SessionMemory"):
        client.collections.create(
            name="SessionMemory",
            properties=[
                Property(name="session_id", data_type=DataType.TEXT),
                Property(name="role", data_type=DataType.TEXT),
                Property(name="user_name", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="timestamp", data_type=DataType.DATE),
            ],
            vectorizer_config=None,
        )

    if not client.collections.exists("WorldState"):
        client.collections.create(
            name="WorldState",
            properties=[
                Property(name="session_id", data_type=DataType.TEXT),
                Property(name="state_json", data_type=DataType.TEXT),
                Property(name="summary", data_type=DataType.TEXT),
                Property(name="version", data_type=DataType.INT),
                Property(name="timestamp", data_type=DataType.DATE),
            ],
            vectorizer_config=None,
        )

ensure_schemas()


class WorldQuery(BaseModel):
    query: str
    user_name: str = "Player"
    session_id: str = "default"
    top_k: int = 5


def _pick_latest_worldstate(ws_objects):
    if not ws_objects:
        return None

    def key(o):
        p = o.properties
        v = p.get("version")
        ts = _ts_to_str(p.get("timestamp"))
        return (v if isinstance(v, int) else -1, ts)

    return sorted(ws_objects, key=key, reverse=True)[0]


def background_update_logic(session_id: str, new_content: str):
    import re
    import json
    print(f"\n🔄 [Background] Updating state for {session_id}...")
    try:
        mem_coll = client.collections.get("SessionMemory")
        ws_coll = client.collections.get("WorldState")
        session_filter = Filter.by_property("session_id").equal(session_id)

        # 1. 取得歷史
        recent = mem_coll.query.fetch_objects(limit=8, filters=session_filter).objects
        # 嘗試找出 User 的名字
        user_name = next((r.properties.get('user_name') for r in recent if r.properties.get('role') == 'user'), "玩家")
        chat_log = "\n".join([f"{r.properties.get('user_name')}: {r.properties.get('text')}" for r in recent])

        ws_objs = ws_coll.query.fetch_objects(limit=1, filters=session_filter).objects
        latest_ws = _pick_latest_worldstate(ws_objs)
        prev_state_json = latest_ws.properties.get("state_json") if latest_ws else "{}"
        prev_version = latest_ws.properties.get("version") if latest_ws else 0

        # 2. 精簡化 Prompt：強調「玩家不屬於 NPC」
        update_prompt = f"""
你是一個世界觀數據提取器。請根據劇情更新 JSON 數據。

【警告】玩家名字是「{user_name}」，「{user_name}」的所有心理與動作嚴禁放入 "npc" 欄位。
"npc" 欄位只紀錄非玩家的異想體或同事。

【結構需求】
{{
  "npc": ["非玩家角色的狀態"],
  "places": ["場景描述"],
  "items": ["物品描述"],
  "events": ["事件摘要"]
}}

【舊狀態】: {prev_state_json}
【新劇情】: {chat_log}\n{new_content}
""".strip()

        raw_res = llm.invoke([
            SystemMessage(content="You are a JSON formatter. Output ONLY valid JSON. No Markdown. Use single double-quotes for keys."),
            HumanMessage(content=update_prompt)
        ]).content

        print("-" * 30 + "\n【DEBUG: LLM 原始回傳】\n" + raw_res + "\n" + "-" * 30)

        # 3. 強力清理：修正雙引號錯誤
        clean_res = raw_res.replace("```json", "").replace("```", "").strip()
        # 將 ""Key"" 替換為 "Key"
        clean_res = re.sub(r'""(\w+)""', r'"\1"', clean_res)

        match = re.search(r'(\{.*\})', clean_res, re.DOTALL)
        if not match: raise ValueError("找不到 JSON 結構")

        json_str = match.group(1)
        json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str) # 移除控制字元

        try:
            state_data = json.loads(json_str)
        except json.JSONDecodeError:
            # 最後一招：移除多餘逗號並修正 key 引號
            json_str = re.sub(r',(\s*[\]\}])', r'\1', json_str)
            json_str = re.sub(r'(?<!")(\b\w+\b)(?!")(?=\s*:)', r'"\1"', json_str)
            state_data = json.loads(json_str)

        new_state_str = json.dumps(state_data, ensure_ascii=False)

        # 4. 文學化敘事摘要 (玩家心理放這裡，而不是 JSON)
        summary_prompt = f"""
請根據以下 JSON 與劇情，撰寫一段 300 字的沉浸式敘事摘要。
這段摘要將作為下次對話的背景參考。
【要求】
1. 深入描寫玩家角色【{user_name}】的心理推論與不安。
2. 描述異想體與環境的壓抑感。
3. 使用繁體中文（臺灣），無標籤、無標題。
數據：{new_state_str}
""".strip()

        summary_res = llm.invoke([HumanMessage(content=summary_prompt)]).content
        summary_res = to_traditional_zh(summary_res)

        ws_coll.data.insert(
            properties={
                "session_id": session_id,
                "state_json": new_state_str,
                "summary": summary_res,
                "version": int(prev_version) + 1,
                "timestamp": utc_now_iso(),
            },
            vector=text_embedder.encode(summary_res)
        )
        print(f"✅ [Background] State updated to v{int(prev_version) + 1}")

    except Exception as e:
        print(f"❌ [Background] Update failed: {e}")

@app.post("/world")
async def chat_world(q: WorldQuery, background_tasks: BackgroundTasks):
    mem_coll = client.collections.get("SessionMemory")
    ws_coll = client.collections.get("WorldState")

    lore_coll = client.collections.get("WorldLoreV2") if client.collections.exists("WorldLoreV2") else None

    session_filter = Filter.by_property("session_id").equal(q.session_id)

    # 1) Save user input
    mem_coll.data.insert(
        properties={
            "session_id": q.session_id,
            "role": "user",
            "user_name": q.user_name,
            "text": q.query,
            "type": "utterance",
            "timestamp": utc_now_iso(),
        },
        vector=text_embedder.encode(q.query),
    )

    # 2.1) World state
    ws_objs = ws_coll.query.fetch_objects(limit=20, filters=session_filter).objects
    latest_ws = _pick_latest_worldstate(ws_objs)
    ws_summary = latest_ws.properties.get("summary") if latest_ws else "初始狀態"
    ws_summary = to_traditional_zh(ws_summary)

    # 2.2) Recent memory
    recent_mem = mem_coll.query.fetch_objects(limit=12, filters=session_filter).objects
    recent_mem = sorted(recent_mem, key=lambda o: _ts_to_str(o.properties.get("timestamp")))
    history_text = "\n".join([
        f"{m.properties.get('user_name') or m.properties.get('role')}: {m.properties.get('text')}"
        for m in recent_mem
    ])

    # 2.3) Lore retrieval (V2)
    lore_text = ""
    used_lore = []
    if lore_coll is not None:
        q_vec = text_embedder.encode(q.query)
        lore_res = lore_coll.query.near_vector(near_vector=q_vec, limit=max(3, q.top_k)).objects

        chunks = []
        for r in lore_res:
            p = r.properties
            name = p.get("name") or "(未命名)"
            ltype = p.get("type") or ""
            txt = p.get("text_zh") or ""
            src = p.get("source_title") or p.get("source_url") or "wiki"
            if txt:
                chunks.append(f"[{ltype}:{name}｜{src}] {txt}")
                used_lore.append({"type": ltype, "name": name, "source": src})
        lore_text = "\n".join(chunks)

    system_prompt = f"""
    你是一位專精於「新本格派懸疑」與「克蘇魯風格」的資深 DM。

    【語言風格：嚴禁翻譯腔（極重要）】
    1. **禁止冗長虛詞**：嚴禁使用「...的事情」、「...的部分」、「...的一種...的感覺」、「進行一個...的動作」。
    2. **禁止萬用動詞**：不要說「感受到壓力」，要說「那股無形的壓迫感正啃噬著你的後頸」。
    3. **台灣文學語感**：使用簡潔、精準、冷峻的繁體中文。避免使用長串的英文式形容詞子句。
    4. **拒絕套話**：刪除「不確定感」、「混合著」、「似乎是」等模糊詞彙。

    【動態敘事要求】
    - **拒絕靜態觀察**：不要只寫「你的視線跟隨」，要描寫「視網膜捕捉到的殘影」或「腳步聲在空曠大廳產生的迴響」。
    - **心理與環境熔接**：將玩家的「安全顧問」背景與 Lore 結合。看到員工喊叫，身為顧問的你，腦中應反射性閃過「收容失效等級」的判斷，而不是只感到不適。
    - **感官細節**：空氣中的惡臭不只是惡臭，那是混合了「消毒水與腐肉」的焦躁氣息。

    【目前情境】
    場景摘要：{ws_summary}
    相關內容（Lore）：{lore_text}
    近期記憶：{history_text}

    【寫作指令】
    請直接從玩家【{q.user_name}】目前的處境推進。請寫出至少 600 字、充滿電影鏡頭感的敘事。
    不要複述玩家的動作，要描述動作產生的「重量」與「後果」。
    嚴禁標註任何「內部單白」或「具象感官」等字眼，將它們融入小說筆觸中。
    """.strip()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=q.query),
    ]).content

    response = light_dedup(to_traditional_zh(response))

    # debug: append citation block (optional)
    if used_lore:
        cites = "；".join([f"{x['type']}:{x['name']}" for x in used_lore[:3]])
        response += f"\n\n【本回合引用】{cites}"

    mem_coll.data.insert(
        properties={
            "session_id": q.session_id,
            "role": "assistant",
            "user_name": "DM",
            "text": response,
            "type": "narrative",
            "timestamp": utc_now_iso(),
        },
        vector=text_embedder.encode(response),
    )

    background_tasks.add_task(background_update_logic, q.session_id, response)

    return {
        "content": response,
        "session_id": q.session_id,
        "debug_info": "State updating in background",
    }


# -----------------
# Debug endpoints v3 (V2)
# -----------------

@app.get("/debug/collections")
def debug_collections(x_debug_token: Optional[str] = Header(default=None, alias="X-Debug-Token")):
    require_debug_token(x_debug_token)
    return {
        "collections": [
            {"name": "WorldLoreV2", "exists": client.collections.exists("WorldLoreV2")},
            {"name": "SessionMemory", "exists": client.collections.exists("SessionMemory")},
            {"name": "WorldState", "exists": client.collections.exists("WorldState")},
        ]
    }


@app.get("/debug/lore")
def debug_lore(
    type: Optional[str] = Query(default=None, description="place/npc/item/rule/event/faction/rumor"),
    tag: Optional[str] = Query(default=None, description="單一 tag 過濾（例如 迷霧）"),
    source_url: Optional[str] = Query(default=None, description="來源網址（只看某篇 wiki 蒸餾）"),
    limit: int = Query(default=20, ge=1, le=200),
    x_debug_token: Optional[str] = Header(default=None, alias="X-Debug-Token"),
):
    try:
        require_debug_token(x_debug_token)

        if not client.collections.exists("WorldLoreV2"):
            return {"count": 0, "items": []}

        lore = client.collections.get("WorldLoreV2")

        filters = None
        if type:
            filters = Filter.by_property("type").equal(type)
        if source_url:
            f2 = Filter.by_property("source_url").equal(source_url)
            filters = f2 if filters is None else (filters & f2)

        res = lore.query.fetch_objects(limit=min(limit, 200), filters=filters).objects

        if tag:
            res = [o for o in res if tag in (o.properties.get("tags") or [])]

        out = []
        for o in res[:limit]:
            p = o.properties
            out.append({
                "uuid": str(o.uuid),
                "card_id": p.get("card_id"),
                "type": p.get("type"),
                "name": p.get("name"),
                "tags": p.get("tags"),
                "text_zh": p.get("text_zh"),
                "source_url": p.get("source_url"),
                "source_lang": p.get("source_lang"),
                "source_title": p.get("source_title"),
                "updated_at": _ts_to_str(p.get("updated_at")),
            })

        return {"count": len(out), "items": out}
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(error_detail) # 這會印在你的終端機視窗
        return {"error": str(e), "detail": "請查看伺服器終端機輸出"}


@app.get("/debug/session")
def debug_session(
    session_id: str = Query(...),
    limit: int = Query(default=30, ge=1, le=200),
    x_debug_token: Optional[str] = Header(default=None, alias="X-Debug-Token"),
):
    require_debug_token(x_debug_token)

    mem = client.collections.get("SessionMemory")
    session_filter = Filter.by_property("session_id").equal(session_id)
    objs = mem.query.fetch_objects(limit=min(limit, 200), filters=session_filter).objects
    objs = sorted(objs, key=lambda o: _ts_to_str(o.properties.get("timestamp")))

    items = []
    for o in objs[-limit:]:
        p = o.properties
        items.append({
            "uuid": str(o.uuid),
            "timestamp": _ts_to_str(p.get("timestamp")),
            "role": p.get("role"),
            "user_name": p.get("user_name"),
            "type": p.get("type"),
            "text": p.get("text"),
        })

    return {"session_id": session_id, "count": len(items), "items": items}


@app.get("/debug/worldstate")
def debug_worldstate(
    session_id: str = Query(...),
    limit: int = Query(default=20, ge=1, le=200),
    x_debug_token: Optional[str] = Header(default=None, alias="X-Debug-Token"),
):
    require_debug_token(x_debug_token)

    ws = client.collections.get("WorldState")
    session_filter = Filter.by_property("session_id").equal(session_id)
    objs = ws.query.fetch_objects(limit=min(limit, 200), filters=session_filter).objects
    latest = _pick_latest_worldstate(objs)

    if not latest:
        return {"session_id": session_id, "exists": False}

    p = latest.properties
    return {
        "session_id": session_id,
        "exists": True,
        "uuid": str(latest.uuid),
        "version": p.get("version"),
        "timestamp": _ts_to_str(p.get("timestamp")),
        "summary": p.get("summary"),
        "state_json": p.get("state_json"),
    }
@app.post("/reset")
def reset_session(session_id: str = Query(...)):
    try:
        # 1. 清理狀態 (WorldState)
        ws_coll = client.collections.get("WorldState")
        ws_coll.data.delete_many(where=Filter.by_property("session_id").equal(session_id))

        # 2. 清理歷史紀錄 (SessionMemory) - 這是解決重複字詞的關鍵
        mem_coll = client.collections.get("SessionMemory")
        mem_coll.data.delete_many(where=Filter.by_property("session_id").equal(session_id))

        return {"msg": f"Session {session_id} has been completely wiped."}
    except Exception as e:
        return {"error": str(e)}
@app.get("/debug/history")
def debug_history(session_id: str = Query(...)):
    mem_coll = client.collections.get("SessionMemory")
    res = mem_coll.query.fetch_objects(
        filters=Filter.by_property("session_id").equal(session_id),
        limit=20
    ).objects

    out = []
    for o in res:
        out.append({
            "role": o.properties.get("role"),
            "content": o.properties.get("content")[:30] + "...", # 只看開頭
            "time": str(o.properties.get("timestamp"))
        })
    return {"count": len(out), "history": out}

@app.on_event("shutdown")
def shutdown():
    client.close()
