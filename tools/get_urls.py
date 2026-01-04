import requests

def get_all_pages_via_api():
    base_url = "https://typemoon.fandom.com/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "max" # 一次抓取最大量 (通常是 500)
    }

    all_links = []

    while True:
        response = requests.get(base_url, params=params).json()
        pages = response.get("query", {}).get("allpages", [])

        for p in pages:
            title = p["title"]
            # 排除非條目內容
            if any(x in title for x in ["Category:", "File:", "Template:", "Module:"]):
                continue

            # 組成完整網址
            url = f"https://typemoon.fandom.com/wiki/{title.replace(' ', '_')}"
            all_links.append(url)

        # 檢查是否還有下一頁 (Pagination)
        if "continue" in response:
            params.update(response["continue"])
        else:
            break

    return all_links

if __name__ == "__main__":
    links = get_all_pages_via_api()
    with open("wiki_urls.txt", "w", encoding="utf-8") as f:
        for l in sorted(links):
            f.write(l + "\n")
    print(f"✅ 透過 API 抓取成功！共 {len(links)} 筆。")