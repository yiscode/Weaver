import weaviate
client = weaviate.connect_to_local()

# 刪除舊的、定義不全的集合
if client.collections.exists("WorldState"):
    client.collections.delete("WorldState")
    print("已刪除舊的 WorldState")

# 重新運行你的 rag_server_memory_v2.py，它會自動建立正確的 schema
client.close()