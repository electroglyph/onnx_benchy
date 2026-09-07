"""Metrics computation + report rendering (table/json/csv).

Primary stats are means over per-batch samples (NOT total/elapsed); the
overall total/elapsed rate is included as a diagnostic. Latency is reported
per document (each batch's time divided by batch size). The table's ± is the
standard error of the mean (SEM); JSON/CSV also carry SD and 95% CI.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics

from . import footer

_CI95_Z = 1.96  # normal approx for 95% CI of the mean


def summarize(stats: dict) -> dict:
    times = stats["times"]
    toks = stats["toks"]
    n = len(times)
    batch_size = stats.get("batch_size", 1) or 1
    # Per-document latency: each batch holds `batch_size` documents
    # (packed chunks in pack mode, one doc per sequence with --no-pack),
    # so divide each batch time evenly across its documents.
    doc_times = [t / batch_size for t in times]
    lat_mean_s = sum(doc_times) / n
    lat_std_s = statistics.stdev(doc_times) if n >= 2 else 0.0
    lat_sem_s = lat_std_s / math.sqrt(n) if n >= 2 else 0.0
    per_batch_tps = [t / dt if dt > 0 else 0.0 for t, dt in zip(toks, times)]
    tps_mean = sum(per_batch_tps) / n
    tps_std = statistics.stdev(per_batch_tps) if n >= 2 else 0.0
    tps_sem = tps_std / math.sqrt(n) if n >= 2 else 0.0
    overall = stats["tokens"] / stats["elapsed_s"] if stats["elapsed_s"] > 0 else 0.0
    out = {
        "batches": stats["batches"],
        "batch_size": batch_size,
        "documents": stats.get("documents", stats["batches"] * batch_size),
        "tokens": stats["tokens"],
        "elapsed_s": stats["elapsed_s"],
        "latency_ms": {
            "mean": lat_mean_s * 1000,
            "std": lat_std_s * 1000,
            "sem": lat_sem_s * 1000,
            "ci95": lat_sem_s * _CI95_Z * 1000,
        },
        "throughput_tps": {
            "mean": tps_mean,
            "std": tps_std,
            "sem": tps_sem,
            "ci95": tps_sem * _CI95_Z,
        },
        "throughput_overall_tps": overall,
        "tokens_counted": "non_pad",
        "epochs": stats.get("epochs", 0),
    }
    if "dropped_batches" in stats:
        out["dropped_batches"] = stats["dropped_batches"]
    if n >= 2:
        import numpy as np

        out["latency_ms"]["p50"] = float(np.percentile(np.array(doc_times) * 1000, 50))
        out["latency_ms"]["p95"] = float(np.percentile(np.array(doc_times) * 1000, 95))
    return out


def fmt_latency(mean_ms: float, std_ms: float, n: int) -> str:
    if mean_ms > 1000:
        return (
            f"{mean_ms / 1000:.2f} s ± {std_ms / 1000:.2f} s"
            if n >= 2
            else f"{mean_ms / 1000:.2f} s ± n/a"
        )
    return (
        f"{mean_ms:.2f} ms ± {std_ms:.2f} ms"
        if n >= 2
        else f"{mean_ms:.2f} ms ± n/a"
    )


def fmt_tps(mean: float, std: float, n: int) -> str:
    if n >= 2:
        return f"{mean:.0f} tok/s ± {std:.0f} tok/s"
    return f"{mean:.0f} tok/s ± n/a"


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(n)


def render_table(config_lines: list[tuple[str, str]], rows: list[dict]) -> str:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    table = Table(show_header=True, header_style="bold")
    table.add_column("Backend")
    table.add_column("Batches", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Elapsed", justify="right")
    table.add_column("Mean latency (per doc)")
    table.add_column("Mean ingest")
    for r in rows:
        m = r["summary"]
        lat = m["latency_ms"]
        tps = m["throughput_tps"]
        # sem added later; fall back to std for summaries built before it existed
        lat_err = lat.get("sem", lat["std"])
        tps_err = tps.get("sem", tps["std"])
        table.add_row(
            Text(r["label"]),
            str(m["batches"]),
            fmt_count(m["tokens"]),
            f"{m['elapsed_s']:.1f}s",
            fmt_latency(lat["mean"], lat_err, m["batches"]),
            fmt_tps(tps["mean"], tps_err, m["batches"]),
        )
    console = Console(width=120)
    with console.capture() as cap:
        console.print("[bold]onnx_benchy run configuration[/bold]")
        for key, val in config_lines:
            # markup=False: paths routinely contain [brackets] (rich would eat them)
            console.print(f"  {key+':':<15} {val}", markup=False)
        console.print()
        console.print(table)
    text = cap.get().rstrip("\n")
    return f"{text}\n{footer()}"


def render_json(config: dict, rows: list[dict]) -> str:
    payload = {
        "config": config,
        "results": [
            {"backend": r["provider"], **r["summary"]} for r in rows
        ],
        "generated_by": footer(),
    }
    return json.dumps(payload, indent=1)


def render_csv(config_lines: list[tuple[str, str]], rows: list[dict]) -> str:
    buf = io.StringIO()
    for key, val in config_lines:
        buf.write(f"# {key}: {val}\n")
    w = csv.writer(buf)
    w.writerow(
        ["backend", "batches", "documents", "tokens", "elapsed_s", "latency_ms_mean",
         "latency_ms_sem", "latency_ms_std", "throughput_tps_mean",
         "throughput_tps_sem", "throughput_tps_std"]
    )
    for r in rows:
        m = r["summary"]
        # .get for back-compat with summaries built before documents/batch_size existed
        batch_size = m.get("batch_size", 1) or 1
        documents = m.get("documents", m["batches"] * batch_size)
        lat = m["latency_ms"]
        tps = m["throughput_tps"]
        lat_sem = lat.get("sem", lat["std"])
        tps_sem = tps.get("sem", tps["std"])
        w.writerow(
            [r["provider"], m["batches"], documents, m["tokens"], f"{m['elapsed_s']:.3f}",
             f"{lat['mean']:.3f}", f"{lat_sem:.3f}", f"{lat['std']:.3f}",
             f"{tps['mean']:.1f}", f"{tps_sem:.1f}", f"{tps['std']:.1f}"]
        )
    buf.write(footer())
    return buf.getvalue().rstrip("\n")
