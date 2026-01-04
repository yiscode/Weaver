import requests
import json
import hashlib
import time
from urllib.parse import unquote

def get_page_info_and_content(title):
    base_url = "https://typemoon.fandom.com/api.php"
    # 使用 query + export 模式，這種模式對特殊字符標題最友善
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions|categories",
        "rvprop": "content", # 抓取原始碼
        "redirects": 1,
        "cllimit": "max"
    }

    try:

        res = requests.get(base_url, params=params).json()
        # 處理重新導向的邏輯
        if "query" in res and "redirects" in res["query"]:
            real_title = res["query"]["redirects"][0]["to"]
            print(f"(Redirect -> {real_title})", end=" ")

        pages = res.get("query", {}).get("pages", {})
        page_id = next(iter(pages))


        if page_id == "-1":
            return None, None

        page_data = pages[page_id]

        # 取得原始 WikiText
        raw_text = page_data.get("revisions", [{}])[0].get("*", "")

        # 簡單清洗：去掉 Wiki 的 [[ ]] 和 {{ }} 標籤
        import re
# --- 改進清洗邏輯 ---
        # 不要直接刪除樣板，改為提取裡面的文字，或者只刪除特定的系統標籤
        clean_text = raw_text
        # 只去掉 Wiki 連結符號，保留裡面的文字
        clean_text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', clean_text)
        # 去掉 ''' (粗體)
        clean_text = clean_text.replace("'''", "").replace("''", "")
        # 去掉 <ref> 標籤
        clean_text = re.sub(r'<ref.*?>.*?</ref>', '', clean_text, flags=re.DOTALL)
        # 去掉 HTML 註釋
        clean_text = re.sub(r'', '', clean_text, flags=re.DOTALL)

        # 判定類型
        categories = [c["title"] for c in page_data.get("categories", [])]
        ltype = "lore"
        # 只要分類裡有 Abnormality，不論大小寫
        cat_str = "|".join(categories).lower()
        if "abnormality" in cat_str or "abnormalities" in cat_str:
            ltype = "abnormality"
        elif "character" in cat_str or "sephirah" in cat_str:
            ltype = "character"

        return clean_text.strip(), ltype
    except Exception as e:
        return None, None

def process_urls_to_jsonl(input_file="wiki_urls.txt", output_file="pm_lore_final.jsonl"):
    count = 0
    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "w", encoding="utf-8") as out:

        for line in f:
            url = line.strip()
            if not url or "/wiki/" not in url: continue

            raw_title = url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ")

            # Debug: 顯示正在嘗試的標題
            print(f"🔍 嘗試抓取: [{title}]", end=" ")

            content, ltype = get_page_info_and_content(title)

            # 放寬限制到 20 個字，因為有些編號頁面真的很短
            if content and len(content) > 20:
                entry = {
                    "id": f"pm_{hashlib.md5(title.encode()).hexdigest()[:8]}",
                    "type": ltype,
                    "name": title,
                    "text_zh": content,
                    "source_url": url,
                    "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
                print(f"-> ✅ 成功 ({ltype})")
            else:
                print("-> ❌ 內容不足")

            time.sleep(0.2)

    print(f"\n✨ 任務完成！共存入 {count} 筆。")

if __name__ == "__main__":
    process_urls_to_jsonl()