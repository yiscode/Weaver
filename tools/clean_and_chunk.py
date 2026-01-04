import json
import re

def clean_wiki_text(text):
    # 1. 移除 Wiki 表格與 tabber 內容
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    text = re.sub(r'<tabber>.*?</tabber>', '', text, flags=re.DOTALL)

    # 2. 處理連結 [[A|B]] -> B
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)

    # 3. 移除 HTML 標籤與內部的註釋
    text = re.sub(r'', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)

    # 4. 移除 Wiki 模板屬性名稱 (例如 |name=, |voice=)
    text = re.sub(r'\|\s*\w+\s*=', ' ', text)

    # 5. 移除格式符號
    text = text.replace("'''", "").replace("''", "").replace('==', '')

    # 6. 暴力清理剩餘的大括號與特殊標點
    text = text.replace('{', '').replace('}', '').replace(';', '').replace('\t', ' ')

    # 7. 移除換行並壓縮空格
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    # 針對 Ruby 標籤的優化：只保留第一個參數（漢字）
    # 範例：{{ruby|十面埋伏|...}} -> 十面埋伏
    text = re.sub(r'\{\{ruby\|([^|]+)\|[^}]+\}\}', r'\1', text)

    return text.strip()

def chunk_text_with_overlap(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    # 如果 text 長度小於 chunk_size，直接回傳
    if len(text) <= chunk_size:
        return [text]

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        # 如果已經處理到結尾，就跳出
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

def chunk_data(input_file="pm_lore_final.jsonl", output_file="DarkSouls.jsonl"):
    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            data = json.loads(line)
            name = data.get("name", "未知主題")
            raw_text = data.get("text_zh", "")

            # 1. 執行深度清洗
            cleaned_text = clean_wiki_text(raw_text)

            # 2. 如果清洗後沒內容就跳過
            if len(cleaned_text) < 10: continue

            # 3. 使用重疊切分
            # 注意：因為換行符被拿掉了，Prefix 也要跟著拿掉換行
            prefix = f"關於【{name}】的設定資料： "

            # 設定切分長度
            chunks = chunk_text_with_overlap(cleaned_text, chunk_size=800, overlap=100)

            for i, chunk in enumerate(chunks):
                new_data = data.copy()
                new_data["id"] = f"{data['id']}_{i}"
                # 注入上下文前綴，保持長字串格式
                new_data["text_zh"] = prefix + chunk
                out.write(json.dumps(new_data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    chunk_data()
    print("✅ 深度清洗完成：已移除所有 { } 與換行符，轉為長字串格式。")