"""One-shot downloader: vendor ~10MB of FineWeb sample-10BT as one-doc-per-line text.

Usage: .venv/bin/python scripts/fetch_fineweb.py [--bytes N] [--out PATH]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os

BUDGET = 10 * 1024 * 1024
SOURCE = "HuggingFaceFW/fineweb"
CONFIG = "sample-10BT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=BUDGET)
    ap.add_argument("--out", default="data/fineweb-10mb.txt")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(SOURCE, name=CONFIG, split="train", streaming=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total, lines, skipped = 0, 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 200:
                skipped += 1
                continue
            line = text.replace("\n", " ") + "\n"
            b = len(line.encode("utf-8"))
            if total + b > args.bytes:
                break
            f.write(line)
            total += b
            lines += 1
    if not 9 * 1024 * 1024 <= total <= 11 * 1024 * 1024:
        raise SystemExit(f"size guard failed: {total} bytes not in 9-11MB")
    with open(args.out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "source": SOURCE,
        "config": CONFIG,
        "split": "train",
        "column": "text",
        "bytes": total,
        "lines": lines,
        "skipped_short": skipped,
        "sha256": digest,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "filter": "len>=200 chars, newlines collapsed, one doc per line, no shuffle",
    }
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"wrote {args.out}: {total/1024/1024:.2f} MiB, {lines} lines")


if __name__ == "__main__":
    main()
