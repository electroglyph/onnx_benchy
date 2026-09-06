"""Tokenizer loading (local dir or HF Hub id)."""

from __future__ import annotations

import os


def load_tokenizer(arg: str, offline: bool, trust_remote_code: bool):
    from transformers import AutoTokenizer

    kwargs: dict = {"trust_remote_code": trust_remote_code, "use_fast": True}
    if offline:
        kwargs["local_files_only"] = True
    try:
        tok = AutoTokenizer.from_pretrained(arg, **kwargs)
    except Exception as e:  # noqa: BLE001 - HF raises many types; all are fatal here
        raise SystemExit(
            f"error: could not load tokenizer {arg!r} "
            f"({'local' if os.path.isdir(arg) else 'hub id'}): {e}\n"
            "hint: pass a local dir (e.g. --tokenizer ./tok/) or a Hub id "
            "(e.g. --tokenizer BAAI/bge-small-en-v1.5); use --offline only "
            "with fully cached/local tokenizers"
        )
    tok.padding_side = "right"
    if tok.pad_token is None:
        for fallback in ("eos_token", "unk_token"):
            if getattr(tok, fallback, None) is not None:
                tok.pad_token = getattr(tok, fallback)
                print(f"warning: no pad_token; falling back to {fallback}")
                break
    if tok.pad_token is None:
        raise SystemExit(
            "error: tokenizer has no pad_token and no eos/unk fallback; "
            "cannot batch. Use a tokenizer with a pad token."
        )
    return tok


def effective_context_size(tokenizer, requested: int) -> tuple[int, str | None]:
    """Clamp --context-size to the tokenizer's model_max_length if smaller."""
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is not None and max_len < 1_000_000 and requested > max_len:
        return int(max_len), (
            f"warning: --context-size {requested} exceeds tokenizer "
            f"model_max_length {max_len}; clamped to {max_len}"
        )
    return requested, None
