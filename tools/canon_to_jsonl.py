import argparse
import json
import hashlib
from datetime import datetime, timezone

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_ws(s: str) -> str:
    return " ".join((s or "").split())


def compute_hash(card: dict) -> str:
    base = "::".join([
        card.get("type", ""),
        card.get("name", ""),
        norm_ws(card.get("text_zh", "")),
        "|".join(card.get("tags", []) or []),
        "|".join(card.get("hooks", []) or []),
        "|".join(card.get("rules", []) or []),
        card.get("source_url", ""),
    ])
    return sha256_text(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="canon.yaml")
    ap.add_argument("--output", default="lore_cards_canon.jsonl")
    ap.add_argument("--source_url", default="canon://myworld")
    ap.add_argument("--source_lang", default="zh")
    ap.add_argument("--source_title", default="Canon")
    ap.add_argument("--license", default="self")
    args = ap.parse_args()

    data = yaml.safe_load(open(args.input, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("canon.yaml must be a YAML list of Lore Cards")

    out_lines = []
    for c in data:
        if not isinstance(c, dict):
            continue

        # Required fields
        cid = c.get("id") or c.get("card_id")
        if not cid:
            # Generate a stable-ish id from name+type+text
            seed = (c.get("type","") + "|" + c.get("name","") + "|" + norm_ws(c.get("text_zh") or c.get("text") or ""))[:2000]
            cid = "canon_" + sha256_text(seed)[:16]

        item = {
            "id": cid,
            "type": c.get("type", "rumor"),
            "name": c.get("name", cid),
            "tags": c.get("tags", []) or [],
            "text_zh": c.get("text_zh") or c.get("text") or "",
            "text_original": c.get("text_original", ""),
            "hooks": c.get("hooks", []) or [],
            "rules": c.get("rules", []) or [],
            "source_url": c.get("source_url") or args.source_url,
            "source_lang": c.get("source_lang") or args.source_lang,
            "source_title": c.get("source_title") or args.source_title,
            "license": c.get("license") or args.license,
        }

        item["content_hash"] = compute_hash(item)
        item["updated_at"] = utc_now_iso()

        out_lines.append(item)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in out_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(out_lines)} cards -> {args.output}")


if __name__ == "__main__":
    main()