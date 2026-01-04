import weaviate
client = weaviate.connect_to_local(port=9080, grpc_port=50051)

# 刪除舊的、定義不全的集合
if client.collections.exists("WorldLoreV2"):
    client.collections.delete("WorldLoreV2")
    print("已刪除舊的 WorldLoreV2")

# 重新運行你的 rag_server_memory_v2.py，它會自動建立正確的 schema
client.close()