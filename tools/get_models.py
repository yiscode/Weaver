import requests

url = "http://localhost:1337/v1/models"
headers = {
    "Authorization": "Bearer my-secret-key" # 這裡填入你的 Key
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    models = response.json()
    for model in models['data']:
        print(f"可用模型 ID: {model['id']}")
except Exception as e:
    print(f"獲取失敗: {e}")