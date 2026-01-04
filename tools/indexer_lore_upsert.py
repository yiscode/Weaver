import json, uuid, re, time, argparse, os
from datetime import datetime, timezone
import logging
import torch

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter, Sort
from sentence_transformers import SentenceTransformer
from opencc import OpenCC
from deep_translator import GoogleTranslator

# === 設定 ===
EMBED_MODEL = "BAAI/bge-m3"
COLL = "WorldLoreV2"
BATCH_SIZE = 64  # 已確認 GPU 可用，建議可調大至 64

# 處理安全檢查報錯 (針對 transformers < 4.48 / torch < 2.6)
try:
    import transformers.utils.import_utils as hf_utils
    hf_utils.check_torch_load_is_safe = lambda: None
except:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
log = logging.getLogger("weaver_indexer_gpu")

t2tw = OpenCC('s2twp')

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def auto_translate_batch(texts):
    """批次翻譯優化"""
    results = []
    translator = GoogleTranslator(source='en', target='zh-TW')
    for text in texts:
        try:
            if not text or not text.strip():
                results.append("")
                continue
            translated = translator.translate(text)
            results.append(t2tw.convert(translated))
        except Exception as e:
            log.error(f"翻譯出錯: {e}")
            results.append(text)
    return results

def detect_tags(text, url):
    tags = []
    url_lower = url.lower()
    if any(k in url_lower for k in ["typemoon", "fate"]):
        tags.append("Fate")
    if any(k in url_lower for k in ["projectmoon", "lobotomy", "limbuscompany"]):
        tags.append("ProjectMoon")
    if any(k in url_lower for k in ["darksouls", "dark-souls", "dark-souls-3", "ds3", "ds3remastered"]):
        tags.append("DarkSouls")
    tags.append("Source_EN")
    return list(set(tags))

def stable_uuid(card_id: str, source_url: str) -> str:
    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(ns, f"{card_id}|{source_url}"))

def ensure_schema(client):
    if not client.collections.exists(COLL):
        client.collections.create(
            name=COLL,
            properties=[
                Property(name="card_id", data_type=DataType.TEXT),
                Property(name="type", data_type=DataType.TEXT),
                Property(name="name", data_type=DataType.TEXT),
                Property(name="tags", data_type=DataType.TEXT_ARRAY),
                Property(name="text_zh", data_type=DataType.TEXT),
                Property(name="text_original", data_type=DataType.TEXT),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="content_hash", data_type=DataType.TEXT),
                Property(name="updated_at", data_type=DataType.DATE),
            ]
        )

def query_latest_project_moon(client):
    """查詢 DarkSouls 標籤的最新 5 筆資料"""
    log.info("🔍 正在查詢 DarkSouls 最新資料...")
    coll = client.collections.get(COLL)

    response = coll.query.fetch_objects(
        filters=Filter.by_property("tags").contains_any(["DarkSouls"]),
        sort=Sort.by_property("updated_at", ascending=False),
        limit=5
    )

    for obj in response.objects:
        p = obj.properties
        print(f"[{p.get('updated_at')}] {p.get('name')} (ID: {p.get('card_id')}) (Tags: {p.get('tags')}) (Content: {p.get('text_zh')[:50]}...)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="pm_lore_cleaned.jsonl")
    ap.add_argument("--query", action="store_true", help="執行查詢而非索引")
    args = ap.parse_args()

    client = weaviate.connect_to_local(port=9080, grpc_port=50051)

    try:
        ensure_schema(client)

        if args.query:
            query_latest_project_moon(client)
            return

        # --- 1. 初始化 GPU 模型 ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"正在載入模型至設備: {device}")
        embedder = SentenceTransformer(EMBED_MODEL, device=device)
        if device == "cuda":
            embedder.half()

        coll = client.collections.get(COLL)

        # 讀取資料
        raw_data_list = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                raw_data_list.append(json.loads(line))

        # --- 2. 預檢：過濾重複 UUID ---
        log.info("🧹 正在檢查重複資料...")
        # 計算所有 UUID
        data_map = {}
        for d in raw_data_list:
            uid = stable_uuid(d.get("id"), d.get("source_url"))
            data_map[uid] = d

        # 一次性查詢已存在的 UUID (分批查詢避免 URL 過長)
        existing_uuids = set()
        all_uids = list(data_map.keys())
        for i in range(0, len(all_uids), 100):
            batch_uids = all_uids[i:i+100]
            # 檢查物件是否存在
            for check_uid in batch_uids:
                if coll.data.exists(check_uid):
                    existing_uuids.add(check_uid)

        to_process_uids = [u for u in all_uids if u not in existing_uuids]
        log.info(f"總筆數: {len(all_uids)}, 已存在: {len(existing_uuids)}, 待處理: {len(to_process_uids)}")

        # --- 3. 批次處理邏輯 ---
        for i in range(0, len(to_process_uids), BATCH_SIZE):
            batch_uids = to_process_uids[i : i + BATCH_SIZE]
            batch_data = [data_map[uid] for uid in batch_uids]

            current_batch_texts = [d.get("text_zh") or d.get("text_original", "") for d in batch_data]
            current_batch_names = [d.get("name", "") for d in batch_data]

            log.info(f"正在處理批次 {i//BATCH_SIZE + 1} ({current_batch_names[0]}...)")

            # 翻譯與向量化
            translated_texts = auto_translate_batch(current_batch_texts)
            vectors = embedder.encode(translated_texts, batch_size=BATCH_SIZE, convert_to_tensor=False)

            # 寫入 Weaviate
            with coll.batch.dynamic() as batch:
                for idx, uid in enumerate(batch_uids):
                    item = batch_data[idx]
                    batch.add_object(
                        uuid=uuid.UUID(uid),
                        properties={
                            "card_id": item.get("id"),
                            "type": item.get("type", ""),
                            "name": item.get("name", ""),
                            "tags": detect_tags(item.get("text_original", ""), item.get("source_url", "")),
                            "text_zh": translated_texts[idx],
                            "text_original": item.get("text_original", ""),
                            "source_url": item.get("source_url", ""),
                            "content_hash": item.get("content_hash", ""),
                            "updated_at": utc_now_iso(),
                        },
                        vector=vectors[idx].tolist()
                    )

            if coll.batch.failed_objects:
                log.error(f"批次寫入失敗筆數: {len(coll.batch.failed_objects)}")

        log.info("✅ GPU 批次處理與 Upsert 完成")

    finally:
        client.close()

if __name__ == "__main__":
    main()