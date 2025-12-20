import requests

# 目標 API URL (FastAPI 伺服器)
url = "http://localhost:9000/world"

# 要送出的 JSON payload
payload = {
    "query": "面前的男人有一種濕滑的腥味，和都城中，那些虔誠的信眾有所區別，這片土地長年以來籠罩在異教的陰影下，黑潮信仰的歷史幾乎可以追溯到第一批移民，我不動聲色的向後挪了挪，問道是什麼事情發生",
    "session_id": "006",
    "top_k": 5,
    "alpha": 0.5,
    "use_rerank": True
}

# 發送 POST 請求
response = requests.post(url, json=payload)

# 顯示回應
print("狀態碼:", response.status_code)
print("內容:", response.text)


try:
    data = response.json()
    print("\n=== 回應內容 ===")
    print("生成敘事:\n", data.get("content", ""))
    print("\n引用來源:", data.get("sources", []))
    print("Session ID:", data.get("session_id", ""))
    print("WorldState 版本:", data.get("state_version", ""))
except Exception:
    print("回應不是 JSON:", response.text)
