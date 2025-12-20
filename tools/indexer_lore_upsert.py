import json, uuid
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
COLL = "WorldLoreV2"


def utc_now_iso() -> str:

    return datetime.now(timezone.utc).isoformat()


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
                Property(name="source_lang", data_type=DataType.TEXT),
                Property(name="source_title", data_type=DataType.TEXT),
                Property(name="license", data_type=DataType.TEXT),
                Property(name="content_hash", data_type=DataType.TEXT),
                Property(name="updated_at", data_type=DataType.DATE),
            ],
            vectorizer_config=None,
        )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="lore_cards.jsonl")
    args = ap.parse_args()

    embedder = SentenceTransformer(EMBED_MODEL)
    client = weaviate.connect_to_local(host="localhost", port=9080, grpc_port=50051, skip_init_checks=True)

    try:
        ensure_schema(client)
        coll = client.collections.get(COLL)

        for line in open(args.input, "r", encoding="utf-8"):
            c = json.loads(line)
            cid = c.get("id")
            url = c.get("source_url", "")
            uid = stable_uuid(cid, url)
            chash = c.get("content_hash", "")

            got = coll.query.fetch_objects(filters=Filter.by_id().equal(uid), limit=1).objects
            if got:
                prev = got[0].properties.get("content_hash")
                if prev == chash:
                    continue
                coll.data.delete_by_id(uid)

            text_zh = c.get("text_zh", "")
            vec = embedder.encode(text_zh)

            coll.data.insert(
                uuid=uid,
                properties={
                    "card_id": cid,
                    "type": c.get("type", ""),
                    "name": c.get("name", ""),
                    "tags": c.get("tags", []) or [],
                    "text_zh": text_zh,
                    "text_original": c.get("text_original", ""),
                    "source_url": url,
                    "source_lang": c.get("source_lang", ""),
                    "source_title": c.get("source_title", ""),
                    "license": c.get("license", ""),
                    "content_hash": chash,
                    "updated_at": utc_now_iso(),
                },
                vector=vec,
            )

        print("✅ Upsert complete")

    finally:
        client.close()


if __name__ == "__main__":
    main()
