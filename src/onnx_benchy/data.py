"""Benchmark corpus loading.

The corpus is one document per line (UTF-8). Small enough (~10MB) to hold
in RAM. Wrap-around with per-epoch reshuffle decouples --tokens from size.
"""

from __future__ import annotations

import random


def load_texts(path: str, shuffle: bool, seed: int) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            texts = [line.strip() for line in f]
    except OSError as e:
        raise SystemExit(
            f"error: could not read --data {path!r}: {e}\n"
            "hint: default is data/fineweb-10mb.txt (run scripts/fetch_fineweb.py "
            "to generate it) or pass --data <your-text-file>"
        )
    texts = [t for t in texts if t]
    if not texts:
        raise SystemExit(f"error: no non-empty lines in --data {path!r}")
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(texts)
    return texts


def reshuffle(texts: list[str], seed: int, epoch: int) -> list[str]:
    rng = random.Random(seed + epoch)
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order]
