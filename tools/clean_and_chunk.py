import json
import re

def clean_wiki_text(text):
    # 1. 去除 HTML 標籤 (如 <span...>)
    text = re.sub(r'<[^>]+>', '', text)
    # 2. 去除模板 {{...}}
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)
    # 3. 處理連結 [[A|B]] -> B, [[A]] -> A
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # 4. 去除多餘的括號和標題符號
    text = text.replace('==', '').replace("'''", "").replace("''", "")
    # 5. 合併多餘換行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_data(input_file="pm_lore_final.jsonl", output_file="pm_lore_cleaned.jsonl"):
    with open(input_file, "r", encoding="utf-8") as f, \
         open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            data = json.loads(line)
            raw_text = data["text_zh"]
            cleaned_text = clean_wiki_text(raw_text)

            # 如果文字太長 (> 1000字)，進行簡單切分
            if len(cleaned_text) > 1000:
                paragraphs = cleaned_text.split('\n\n')
                current_chunk = ""
                chunk_id = 0
                for p in paragraphs:
                    if len(current_chunk) + len(p) < 1000:
                        current_chunk += p + "\n\n"
                    else:
                        new_data = data.copy()
                        new_data["id"] = f"{data['id']}_{chunk_id}"
                        new_data["text_zh"] = current_chunk.strip()
                        out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                        current_chunk = p + "\n\n"
                        chunk_id += 1
                if current_chunk:
                    new_data = data.copy()
                    new_data["id"] = f"{data['id']}_{chunk_id}"
                    new_data["text_zh"] = current_chunk.strip()
                    out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
            else:
                data["text_zh"] = cleaned_text
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    chunk_data()
    print("✅ 清洗與切分完成！現在可以使用 pm_lore_cleaned.jsonl 進行索引了。")