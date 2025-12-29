import json, hashlib
from typing import Dict, Any, List
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

JAN_BASE_URL = "http://localhost:1337/v1"
JAN_API_KEY = "my-secret-key"
JAN_MODEL = "gemma-2-9b-it-abliterated-IQ4_XS"  # 你指定


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_llm():
    return ChatOpenAI(
        base_url=JAN_BASE_URL,
        api_key=JAN_API_KEY,
        model=JAN_MODEL,
        temperature=0.4,
        max_tokens=4096,
    )


def distill_one(llm, raw: Dict[str, Any], n_cards: int) -> List[Dict[str, Any]]:
    url = raw.get("url", "")
    lang = raw.get("lang", "")
    title = raw.get("title", "")
    text = raw.get("text", "")

    system = SystemMessage(content=(
            "你是世界觀資料蒸餾器。請把內容轉為 Lore Cards。\n"
            "【輸出規範】你必須輸出嚴格遵循 JSON 標準的格式：\n"
            "1. 所有的鍵名和字串值都必須使用「雙引號」括起來。\n"
            "2. 嚴禁省略引號（例如禁止 id:1，必須是 \"id\":1）。\n"
            "3. 只輸出 JSON 陣列，不要有任何其他文字。\n"
            "範例格式：[{\"id\": \"wiki_01\", \"type\": \"npc\", \"name\": \"克蘇魯\"}]"
        ))

    prompt = HumanMessage(content=(
        f"請從以下來源蒸餾出 {n_cards} 張 Lore Cards：\n"
        f"來源：{title} ({lang}) {url}\n"
        f"正文：\n{text[:10000]}" # 縮減一點長度確保穩定
    ))

    try:
        response = llm.invoke([system, prompt])
        out = response.content
    except Exception as e:
        print(f"❌ LLM 呼叫失敗: {e}")
        return []

    if not out or not out.strip():
        return []

    # 強效 JSON 提取
    clean_json_str = ""
    match = re.search(r'\[\s*\{.*\}\s*\]', out, re.DOTALL) # 專門抓取 JSON 陣列
    if match:
        clean_json_str = match.group(0)
    else:
        # 退而求其次的清理
        clean_json_str = out.replace("```json", "").replace("```", "").strip()

    try:
        cards = json.loads(clean_json_str)

        # 確保 cards 是清單
        if not isinstance(cards, list):
            cards = [cards] if isinstance(cards, dict) else []

        # 欄位補齊與 Hash 計算
        final_cards = []
        for c in cards:
            if not isinstance(c, dict): continue

            c.setdefault("name", "未命名項目")
            c.setdefault("id", f"wiki_{sha256_text(url + '|' + c.get('name',''))[:16]}")
            c.setdefault("tags", [])
            c.setdefault("hooks", [])
            c.setdefault("rules", [])
            c["source_url"] = url
            c["source_lang"] = lang
            c["source_title"] = title
            c["license"] = "CC BY-SA / GFDL (check source)"

            base_info = (c.get("type",""), c.get("name",""), c.get("text_zh",""))
            c["content_hash"] = sha256_text("::".join(map(str, base_info)))
            final_cards.append(c)

        return final_cards

    except json.JSONDecodeError:
        print(f"❌ 無法解析 JSON。原始輸出片段: {out[:100]}...")
        return []

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="raw_sources.jsonl")
    ap.add_argument("--output", default="lore_cards.jsonl")
    ap.add_argument("--cards", type=int, default=12)
    args = ap.parse_args()

    llm = build_llm()
    with open(args.output, "w", encoding="utf-8") as fo:
        for line in open(args.input, "r", encoding="utf-8"):
            raw = json.loads(line)
            cards = distill_one(llm, raw, n_cards=args.cards)
            for c in cards:
                fo.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ Wrote: {args.output}")


if __name__ == "__main__":
    main()
