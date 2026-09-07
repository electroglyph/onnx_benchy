"""Pre-tokenization + batch building.

Tokenization happens ONCE before timing starts, so reported latency /
throughput = model inference + pooling/norm only ("ingest speed").
Latency is reported per document (batch time / batch size). Corpus
wrap-around (reshuffle + retokenize) runs untimed in the benchmark loop,
so epoch boundaries never spike a sample.

Packing (default): concatenate all token ids, slice into dense
context-size blocks. Every batch is [B, C] non-pad tokens; tok/s is honest.
--no-pack: per-doc truncate+pad to context-size (preserves doc boundaries);
throughput counts non-pad tokens (attention-mask sum) only.
"""

from __future__ import annotations

import numpy as np

from .data import reshuffle


def pre_tokenize(tokenizer, texts: list[str], pack: bool) -> list[list[int]]:
    # verbose=False: long lines always exceed model_max_length here; they are
    # chunked to --context-size below, so the length warning would be noise.
    if pack:
        ids: list[int] = []
        for t in texts:
            ids.extend(tokenizer(t, add_special_tokens=False, verbose=False)["input_ids"])
        return [ids]  # chunked into context-size blocks by BatchStream._prepare
    seqs = []
    for t in texts:
        seqs.append(
            tokenizer(t, add_special_tokens=True, truncation=False,
                      verbose=False)["input_ids"]
        )
    return seqs


class BatchStream:
    """Infinite batch iterator with epoch wrap-around + reshuffle."""

    def __init__(
        self,
        tokenized: list[list[int]],
        batch_size: int,
        context_size: int,
        pack: bool,
        pad_id: int,
        texts: list[str],
        tokenizer,
        shuffle: bool,
        seed: int,
    ):
        self.batch_size = batch_size
        self.context_size = context_size
        self.pack = pack
        self.pad_id = pad_id
        self.texts = texts
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._blocks: list[list[int]] = []
        self._pos = 0
        self._prepare(tokenized)

    def _prepare(self, tokenized: list[list[int]]) -> None:
        if self.pack:
            flat: list[int] = [i for seq in tokenized for i in seq]
            c = self.context_size
            n_full = len(flat) // c
            if n_full == 0:
                raise SystemExit(
                    "error: corpus too small to fill one packed "
                    f"{self.batch_size}x{self.context_size} batch; use a smaller "
                    "--context-size/--batch-size or a bigger --data file"
                )
            flat = flat[: n_full * c]
            self._blocks = [flat[i * c : (i + 1) * c] for i in range(n_full)]
        else:
            c = self.context_size
            self._blocks = [seq[:c] for seq in tokenized]
        self._pos = 0

    def _next_epoch(self) -> None:
        self.epoch += 1
        texts = (
            reshuffle(self.texts, self.seed, self.epoch) if self.shuffle else self.texts
        )
        self._prepare(pre_tokenize(self.tokenizer, texts, self.pack))

    def next_batch(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Returns (input_ids [B,C] int64, attention_mask [B,C] int64, nonpad_count)."""
        b, c = self.batch_size, self.context_size
        ids = np.full((b, c), self.pad_id, dtype=np.int64)
        mask = np.zeros((b, c), dtype=np.int64)
        for r in range(b):
            if self._pos >= len(self._blocks):
                self._next_epoch()
            seq = self._blocks[self._pos]
            self._pos += 1
            n = min(len(seq), c)
            ids[r, :n] = seq[:n]
            mask[r, :n] = 1
        nonpad = int(mask.sum())
        return ids, mask, nonpad
