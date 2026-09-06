"""Timed benchmark loop: warmup (untimed) + timed loop with stop conditions."""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np
import onnxruntime as ort

from .pooling import pool_and_norm
from .progress import bar


def make_session(model_path: str, provider: str):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=opts, providers=[provider])


def run_backend(
    sess,
    stream,
    feed_names: list[str],
    output_index: int,
    pooling: str,
    normalize: bool,
    warmup_batches: int,
    tokens_limit: int | None,
    minutes_limit: float | None,
    label: str,
    show_progress: bool,
) -> dict:
    """Run warmup + timed loop for one backend. Returns stats dict."""
    output_name = sess.get_outputs()[output_index].name

    def one_batch():
        input_ids, mask, nonpad = stream.next_batch()
        feed = {}
        if "input_ids" in feed_names:
            feed["input_ids"] = input_ids
        if "attention_mask" in feed_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in feed_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        outs = sess.run([output_name], feed)
        _emb = pool_and_norm(outs[0], mask, pooling, normalize)
        return nonpad

    if warmup_batches > 0:
        with bar(
            show_progress,
            total=warmup_batches,
            desc=f"warmup ({label})",
            unit="batch",
            leave=False,
        ) as pbar:
            for _ in range(warmup_batches):
                one_batch()
                pbar.update(1)

    times: list[float] = []
    toks: list[int] = []
    tokens_done = 0
    minutes_s = minutes_limit * 60 if minutes_limit is not None else None
    t0 = perf_counter()
    with bar(
        show_progress,
        total=tokens_limit,
        desc=f"benchmark ({label})",
        unit="tok",
    ) as pbar:
        while True:
            s = perf_counter()
            nonpad = one_batch()
            dt = perf_counter() - s
            times.append(dt)
            toks.append(nonpad)
            tokens_done += nonpad
            pbar.update(nonpad)
            elapsed = perf_counter() - t0
            if tokens_limit is not None and tokens_done >= tokens_limit:
                break
            if minutes_s is not None and elapsed >= minutes_s:
                break
    elapsed = perf_counter() - t0
    return {
        "batches": len(times),
        "batch_size": stream.batch_size,
        "documents": len(times) * stream.batch_size,
        "tokens": tokens_done,
        "elapsed_s": elapsed,
        "times": times,
        "toks": toks,
        "epochs": stream.epoch,
    }
