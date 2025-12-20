import re, time, json
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup

UA = "LoreCrawler/1.0 (local use)"

@dataclass
class RawDoc:
    url: str
    lang: str
    title: str
    text: str


def _detect_lang_from_url(url: str) -> str:
    m = re.match(r"https?://([a-z\-]+)\.wikipedia\.org/", url)
    return m.group(1) if m else "unknown"


def fetch_wikipedia_article(url: str, timeout: int = 30) -> RawDoc:
    lang = _detect_lang_from_url(url)
    headers = {"User-Agent": UA}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "lxml")
    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

    content = soup.select_one("#mw-content-text")
    if not content:
        text = soup.get_text("\n", strip=True)
        return RawDoc(url=url, lang=lang, title=title, text=text)

    for sel in [".infobox", ".toc", ".mw-editsection", ".reference", "sup.reference", "table", "style", "script"]:
        for node in content.select(sel):
            node.decompose()

    paras = [p.get_text(" ", strip=True) for p in content.find_all(["p"]) if p.get_text(strip=True)]
    text = "\n".join(paras)
    return RawDoc(url=url, lang=lang, title=title, text=text)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="urls.json: {\"urls\":[...]} 可混合多語言 wiki")
    ap.add_argument("--output", default="raw_sources.jsonl")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()

    data = json.load(open(args.input, "r", encoding="utf-8"))
    urls: List[str] = data.get("urls", [])

    with open(args.output, "w", encoding="utf-8") as f:
        for url in urls:
            doc = fetch_wikipedia_article(url)
            f.write(json.dumps({
                "url": doc.url,
                "lang": doc.lang,
                "title": doc.title,
                "text": doc.text,
                "retrieved_at": int(time.time()),
            }, ensure_ascii=False) + "\n")
            time.sleep(args.sleep)

    print(f"✅ Wrote: {args.output}")


if __name__ == "__main__":
    main()
