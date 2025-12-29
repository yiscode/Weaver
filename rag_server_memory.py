import os, json, hashlib, re, time
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, Header
from pydantic import BaseModel

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# === 配置設定 ===
LLM_URL = os.getenv("LLM_URL", "http://localhost:1337/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma-2-9b-it-abliterated-IQ4_XS")
DEBUG_TOKEN = "weaver_admin_2025"

# Weaviate 連線
client = weaviate.connect_to_local(
    host=os.getenv("WEAVIATE_HOST", "localhost"),
    port=9080,
    grpc_port=50051,
    skip_init_checks=True,
)

print(f"🚀 Weaver Engine 啟動：因果與邏輯強化版 (Model: {LLM_MODEL})")
text_embedder = SentenceTransformer("BAAI/bge-m3")

# 調優模型參數，增加隨機性與懲罰重複
llm = ChatOpenAI(
    base_url=LLM_URL,
    api_key="my-secret-key",
    model=LLM_MODEL,
    temperature=0.75,      # 提高溫度以增加 NPC 的反應不可測性
    max_tokens=650,
    top_p=0.9,
    frequency_penalty=1.4, # 強力防止模型重複玩家或自己的話
    presence_penalty=1.1,
)

app = FastAPI()

# --- 核心工具 ---

def to_traditional_zh(text: str) -> str:
    try:
        from opencc import OpenCC
        return OpenCC('s2twp').convert(text)
    except: return text

def robust_json_decode(s: str):
    """針對 JSON 損壞進行暴力修復與 Regex 備援"""
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if not match: return None
    clean_s = match.group(0).replace("'", '"')
    clean_s = re.sub(r'\n', ' ', clean_s)
    clean_s = re.sub(r',\s*\}', '}', clean_s)
    try:
        return json.loads(clean_s)
    except:
        sum_m = re.search(r'"summary":\s*"([^"]*)"', clean_s)
        return {"summary": sum_m.group(1) if sum_m else "劇情推進中", "npc_names": []}

# --- 身分與存檔管理 ---

def _generate_binding_key(session_id: str, player_id: str) -> str:
    return hashlib.md5(f"{session_id}_{player_id}".encode()).hexdigest()

def _get_pc_name(session_id: str, binding_key: str) -> Optional[str]:
    reg = client.collections.get("PCRegistry")
    f = Filter.by_property("session_id").equal(session_id) & Filter.by_property("player_key").equal(binding_key)
    objs = reg.query.fetch_objects(limit=1, filters=f).objects
    return objs[0].properties.get("pc_name") if objs else None

def _upsert_pc_name(session_id: str, binding_key: str, pc_name: str):
    reg = client.collections.get("PCRegistry")
    f = Filter.by_property("player_key").equal(binding_key)
    reg.data.delete_many(where=f)
    reg.data.insert(properties={
        "session_id": session_id, "player_key": binding_key, "pc_name": pc_name, "timestamp": datetime.now(timezone.utc).isoformat()
    })

# --- 背景同步邏輯 (因果更新) ---

def background_update_logic(session_id: str, new_content: str, pc_name: str, user_query: str, binding_key: str):
    time.sleep(2)
    try:
        ws_coll = client.collections.get("WorldState")
        ws_objs = ws_coll.query.fetch_objects(limit=1, filters=Filter.by_property("session_id").equal(session_id)).objects
        prev_version = ws_objs[0].properties.get("version") if ws_objs else 0

        # 要求模型總結最新狀態
        update_prompt = f"將敘事事實提取為極簡 JSON（繁體中文）。內容：{new_content}"
        raw_res = llm.invoke([
            SystemMessage(content="Return ONLY JSON: {\"npc_names\": [], \"summary\": \"\"}"),
            HumanMessage(content=update_prompt)
        ]).content

        state_data = robust_json_decode(raw_res)
        if state_data:
            ws_coll.data.insert(properties={
                "session_id": session_id,
                "state_json": json.dumps(state_data, ensure_ascii=False),
                "summary": to_traditional_zh(state_data.get("summary", "劇情更新")),
                "version": int(prev_version) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    except Exception as e:
        print(f"⚠️ [Memory Sync] Skip: {e}")

# --- API 進入點 (核心邏輯強化) ---

class WorldQuery(BaseModel):
    query: str
    user_name: str = "Player"
    player_id: Optional[str] = None
    session_id: str = "default"

@app.post("/world")
async def chat_world(q: WorldQuery, background_tasks: BackgroundTasks):
    ws_coll = client.collections.get("WorldState")
    session_filter = Filter.by_property("session_id").equal(q.session_id)

    # 1. 識別身分
    binding_key = _generate_binding_key(q.session_id, q.player_id or q.user_name)
    bracket_match = re.search(r"[\[【](.*?)[\]】]", q.query)
    if bracket_match:
        pc_name = bracket_match.group(1).strip()
        _upsert_pc_name(q.session_id, binding_key, pc_name)
    else:
        pc_name = _get_pc_name(q.session_id, binding_key) or "主角"

    # 2. 獲取背景
    ws_objs = ws_coll.query.fetch_objects(limit=1, filters=session_filter).objects
    ws_summary = to_traditional_zh(ws_objs[0].properties.get("summary")) if ws_objs else "故事剛開始"
    print(ws_summary)

    # 3. 強化 Lore 檢索
    lore_text = ""
    detected_style = "泛用冒險"
    lore_coll = client.collections.get("WorldLoreV2") if client.collections.exists("WorldLoreV2") else None
    if lore_coll:
        search_str = f"{ws_summary} {q.query}"
        lore_res = lore_coll.query.near_vector(near_vector=text_embedder.encode(search_str), limit=3).objects
        if lore_res:
            lore_text = "\n".join([f"- 設定內容：{r.properties.get('text_zh')}" for r in lore_res])
            first_lore = lore_res[0].properties.get('text_zh', '')
            if any(k in first_lore for k in ["公司", "收容", "異想體"]): detected_style = "腦葉驚悚"
            if any(k in first_lore for k in ["神祕", "石頭", "孩子"]): detected_style = "超現實神祕"

    # 4. 因果指令設計 (針對你的問題特別強化)
    causal_logic = f"""
玩家目前的動作或問題：『{q.query}』。
目前所處的情境摘要：『{ws_summary}』。

## 寫作引導方針：
- **拒絕回音**：禁止重新描述玩家已經說過的話（例如：主角詢問了石頭的意思...）。
- **因果反饋**：直接描寫該動作產生的結果、NPC 的回答或環境的突變。
- **資訊增量**：根據參考 Lore 或風格邏輯，提供一個玩家尚未知曉的新線索。
- **感官層次**：描述石頭的溫度變化、孩子的微表情、或房間中某種詭異的寂靜。
""".strip()

    system_prompt = f"""
# 角色設定：【{pc_name}】
# 當前風格：【{detected_style}】
# 指令協議：
1. **直接推進**：跳過鋪陳，從動作造成的【後果】開始敘事。
2. **語系限制**：純【繁體中文】，嚴禁中英夾雜，嚴禁使用「你、我」。
3. **字數**：300-400 字，注重心理與對話。
4. **建議行動**：結尾提供 3 個具備決策意義的選項。

## 邏輯核心：
{causal_logic}

## 參考 Lore：
{lore_text if lore_text else "無特定 Lore。請根據風格產出合乎邏輯的新情節。"}
""".strip()

    # 5. 生成內容
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"玩家行動：{q.query}")
    ]).content

    response = to_traditional_zh(response)

    # 6. 背景任務
    background_tasks.add_task(background_update_logic, q.session_id, response, pc_name, q.query, binding_key)

    return {"content": response, "pc_name": pc_name}

# --- 維護接口 ---
@app.post("/reset")
def reset_session(session_id: str = Query(...)):
    for col in ["WorldState", "SessionMemory", "PCRegistry"]:
        client.collections.get(col).data.delete_many(where=Filter.by_property("session_id").equal(session_id))
    return {"msg": f"Session {session_id} reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9527)