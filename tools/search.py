import argparse
import logging
import weaviate
from sentence_transformers import SentenceTransformer

# === 設定 Logging ===
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag_test")

def run_test_query(query_text, limit=3, collection_name="WorldLoreV2"):
    # 1. 初始化模型 (與 Server 保持一致)
    log.info(f"正在載入模型 BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")

    # 2. 連線 Weaviate
    log.info(f"正在連線 Weaviate...")
    client = weaviate.connect_to_local(
        host="localhost",
        port=9080,
        grpc_port=50051
    )

    try:
        if not client.collections.exists(collection_name):
            log.error(f"錯誤：集合 '{collection_name}' 不存在！")
            return

        # 3. 將查詢文字轉為向量
        log.info(f"正在向量化查詢語句: '{query_text}'")
        query_vector = model.encode(query_text)

        # 4. 執行向量檢索 (Near Vector)
        collection = client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_properties=["name", "text_zh", "type"],
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
        )

        # 5. 印出結果
        log.info("\n" + "="*50)
        log.info(f"🔍 語義檢索結果 (Top {limit}):")
        log.info("="*50)

        if not response.objects:
            log.info("未找到相關結果。")

        for i, obj in enumerate(response.objects):
            props = obj.properties
            dist = obj.metadata.distance
            # 距離越小（接近 0）代表語義越接近
            log.info(f"[{i+1}] 相似度分數 (Distance): {dist:.4f}")
            log.info(f"📌 名稱: {props.get('name')}")
            log.info(f"🏷️ 類型: {props.get('type')}")
            log.info(f"📝 內容摘要: {props.get('text_zh')[:150]}...")
            log.info("-" * 30)

    finally:
        client.close()
        log.info("連線已關閉。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="測試 RAG 語義搜尋")
    parser.add_argument("--input", type=str, required=True, help="輸入要查詢的句子 (例如: '關於燕青的寶具')")
    parser.add_argument("--limit", type=int, default=3, help="回傳結果數量")
    parser.add_argument("--col", type=str, default="WorldLoreV2", help="Weaviate 集合名稱")

    args = parser.parse_args()

    run_test_query(args.input, args.limit, args.col)