import requests

url = "http://localhost:1337/v1/chat/completions"
url1 = "http://localhost:1337/v1/models"


headers = {
    "Authorization": "Bearer my-secret-key",
    "Content-Type": "application/json"
}

payload = {
    "model": "gemma-2-9b-it-abliterated-IQ4_XS",
    "messages": [
        {"role": "system", "content": "你是一位黑暗奇幻敘事者"},
        {"role": "user", "content": "請寫一段關於黑暗儀式的故事"}
    ],
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.95
}

response1 = requests.get(url1, headers=headers)
print(response1.text)
response = requests.post(url, headers=headers, json=payload)

print("狀態碼:", response.status_code)
try:
    data = response.json()
    print("\n=== 回應內容 ===")
    # OpenAI 格式的回應通常在 choices[0].message.content
    print(data["choices"][0]["message"]["content"])
except Exception:
    print("回應不是 JSON:", response.text)
