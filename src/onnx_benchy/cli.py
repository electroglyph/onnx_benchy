"""CLI: argument parsing + multi-backend benchmark orchestration."""

from __future__ import annotations

import argparse
import os
import sys

import onnxruntime as ort

from .backends import FLAG_TO_PROVIDER, PROVIDER_SHORT, resolve_backends
from .batching import BatchStream, pre_tokenize
from .data import load_texts
from .metrics import render_csv, render_json, render_table, summarize
from .model_inspect import (
    ConfigError,
    build_feed_names,
    detect_pooling_normalize,
    format_outputs,
    inspect_model,
    output_rank,
    resolve_output,
)
from .runner import make_session, run_backend
from .tokenizer_loader import effective_context_size, load_tokenizer

DEFAULT_TOKENS = 100_000
DEFAULT_MINUTES = 2.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="onnx_benchy",
        description="Benchmark ONNX embedding models across ONNX Runtime backends.",
    )
    p.add_argument("model", nargs="?", help="path to .onnx file")
    p.add_argument("--tokenizer", required=False, default=None,
                   help="HF id or local dir, e.g. BAAI/bge-small-en-v1.5 or ./tok/")
    p.add_argument("--config-dir", default=None,
                   help="dir with modules.json / *Pooling/config.json "
                        "(default: parent dir of MODEL)")

    g = p.add_argument_group("backends (none => all SUPPORTED available)")
    for flag in FLAG_TO_PROVIDER:
        g.add_argument(f"--{flag}", action="store_true",
                       help=f"use {FLAG_TO_PROVIDER[flag]}")
    g.add_argument("--all", action="store_true", help="explicit auto mode")
    g.add_argument("--list-backends", action="store_true",
                   help="print available providers and exit")

    g = p.add_argument_group("shape")
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--context-size", "--seq-len", "--max-length",
                   dest="context_size", type=int, default=512)
    g.add_argument("--output", default="auto",
                   help="auto | index | output name")
    g.add_argument("--pooling", default="auto",
                   choices=["auto", "cls", "mean", "max", "lasttoken", "last", "none"])
    g.add_argument("--normalize", default="auto", choices=["auto", "true", "false"])
    g.add_argument("--warmup-batches", type=int, default=2)
    g.add_argument("--list-outputs", action="store_true",
                   help="print model outputs and exit")

    g = p.add_argument_group("limits (first hit wins)")
    g.add_argument("--tokens", type=int, default=None)
    g.add_argument("--minutes", type=float, default=None)

    g = p.add_argument_group("misc")
    g.add_argument("--data", default="data/fineweb-10mb.txt")
    g.add_argument("--seed", type=int, default=23)
    g.add_argument("--no-shuffle", action="store_true")
    g.add_argument("--no-pack", action="store_true")
    g.add_argument("--offline", action="store_true")
    g.add_argument("--trust-remote-code", action="store_true")
    g.add_argument("--output-format", default="table", choices=["table", "json", "csv"])
    g.add_argument("--output-file", default=None)
    g.add_argument("--no-progress", action="store_true")
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def list_backends() -> int:
    available = ort.get_available_providers()
    _, _, ignored = resolve_backends([], available)
    print("available providers:")
    for prov in available:
        tag = "ignored (unsupported)" if prov in ignored else "supported"
        print(f"  {prov} [{tag}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_backends:
        return list_backends()
    if not args.model:
        print("error: MODEL path is required\n"
              "usage: onnx_benchy MODEL --tokenizer ID_OR_PATH [options]",
              file=sys.stderr)
        return 2
    if not args.tokenizer:
        print("error: --tokenizer is required (ONNX has no tokenizer)\n"
              "examples: --tokenizer BAAI/bge-small-en-v1.5 | --tokenizer ./tok/",
              file=sys.stderr)
        return 2
    if not os.path.isfile(args.model):
        print(f"error: model file not found: {args.model}", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.context_size < 8 or args.warmup_batches < 0:
        print("error: need --batch-size >= 1, --context-size >= 8, "
              "--warmup-batches >= 0", file=sys.stderr)
        return 2
    if args.tokens is not None and args.tokens < 1:
        print("error: --tokens must be >= 1", file=sys.stderr)
        return 2
    if args.minutes is not None and args.minutes <= 0:
        print("error: --minutes must be > 0", file=sys.stderr)
        return 2

    tokens_limit = args.tokens
    minutes_limit = args.minutes
    if tokens_limit is None and minutes_limit is None:
        tokens_limit, minutes_limit = DEFAULT_TOKENS, DEFAULT_MINUTES

    pooling_req = "lasttoken" if args.pooling == "last" else args.pooling

    if args.config_dir is not None and not os.path.isdir(args.config_dir):
        print(f"error: --config-dir not found: {args.config_dir}", file=sys.stderr)
        return 2

    # --- model inspection (before tokenizer: --list-outputs needs no tokenizer) ---
    try:
        info = inspect_model(args.model)
    except Exception as e:  # noqa: BLE001 - ORT load can fail many ways; all fatal
        print(f"error: could not load model {args.model!r}: {e}", file=sys.stderr)
        return 2
    if args.list_outputs:
        print(format_outputs(info))
        return 0
    try:
        out_idx, out_name = resolve_output(info["outputs"], args.output)
        rank = output_rank(info["outputs"], out_idx)
        config_dir = args.config_dir or os.path.dirname(os.path.abspath(args.model))
        pooling, pooling_src, normalize, normalize_src = detect_pooling_normalize(
            pooling_req, args.normalize, rank, config_dir)
        feed_names = build_feed_names([i["name"] for i in info["inputs"]])
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # --- tokenizer ---
    try:
        tok = load_tokenizer(args.tokenizer, args.offline, args.trust_remote_code)
    except SystemExit as e:
        print(e.code, file=sys.stderr)
        return 2
    context_size, clamp_warn = effective_context_size(tok, args.context_size)
    if clamp_warn:
        print(clamp_warn)

    # --- backends ---
    available = ort.get_available_providers()
    requested = [f for f in FLAG_TO_PROVIDER if getattr(args, f)] or (
        ["all"] if args.all else [])
    resolved, warnings, _ignored = resolve_backends(requested, available)
    for w in warnings:
        print(f"warning: {w}")
    if not resolved:
        print("error: no supported backends available to benchmark "
              f"(available: {available})", file=sys.stderr)
        return 2

    # --- data (tokenize once, reuse across backends) ---
    shuffle = not args.no_shuffle
    pack = not args.no_pack
    try:
        texts = load_texts(args.data, shuffle, args.seed)
    except SystemExit as e:
        print(e.code, file=sys.stderr)
        return 2
    tokenized = pre_tokenize(tok, texts, pack)
    pad_id = tok.pad_token_id
    try:  # fail fast: a shape the corpus can't fill breaks every backend equally
        BatchStream(tokenized, args.batch_size, context_size, pack,
                    pad_id, texts, tok, shuffle, args.seed)
    except SystemExit as e:
        print(e.code, file=sys.stderr)
        return 2

    show_progress = (
        not args.quiet and not args.no_progress and sys.stdout.isatty()
    )

    # --- config echo block ---
    req_backends = ",".join(requested) if requested else "(auto)"
    res_backends = ",".join(PROVIDER_SHORT.get(p, p) for p in resolved)
    config_lines = [
        ("model", args.model),
        ("tokenizer", args.tokenizer),
        ("config-dir", f"{config_dir}"
         + ("" if args.config_dir else " (auto: parent of model)")),
        ("backends", f"{res_backends} (requested: {req_backends})"),
        ("batch-size", str(args.batch_size)),
        ("context-size", str(context_size)
         + (f" (clamped from {args.context_size})" if clamp_warn else "")),
        ("output", f"{out_name} (resolved from: {args.output})"),
        ("pooling", f"{pooling} (requested: {pooling_req}; {pooling_src})"),
        ("normalize", f"{str(normalize).lower()} (requested: {args.normalize}; {normalize_src})"),
        ("warmup-batches", str(args.warmup_batches)),
        ("tokens", str(tokens_limit) if tokens_limit is not None else "inf"),
        ("minutes", str(minutes_limit) if minutes_limit is not None else "inf"),
        ("data", args.data),
        ("seed", f"{args.seed} ({'no shuffle' if args.no_shuffle else 'shuffle on'})"),
        ("packing", "no-pack (doc boundaries)" if args.no_pack else "pack (dense chunks)"),
        ("offline", str(args.offline).lower()),
        ("trust-remote-code", str(args.trust_remote_code).lower()),
    ]
    config_json = {
        "model": args.model, "tokenizer": args.tokenizer,
        "config_dir": config_dir,
        "backends_requested": requested, "backends_resolved": resolved,
        "batch_size": args.batch_size, "context_size": context_size,
        "output_requested": args.output, "output_resolved": out_name,
        "pooling_requested": pooling_req, "pooling_resolved": pooling,
        "pooling_source": pooling_src,
        "normalize_requested": args.normalize, "normalize_resolved": normalize,
        "normalize_source": normalize_src,
        "warmup_batches": args.warmup_batches,
        "tokens": tokens_limit, "minutes": minutes_limit,
        "data": args.data, "seed": args.seed, "shuffle": shuffle,
        "packing": "no-pack" if args.no_pack else "pack",
        "offline": args.offline, "trust_remote_code": args.trust_remote_code,
    }

    # --- per-backend runs ---
    rows: list[dict] = []
    interrupted = False
    for provider in resolved:
        label = PROVIDER_SHORT.get(provider, provider)
        try:
            sess = make_session(args.model, provider)
        except Exception as e:  # noqa: BLE001 - any session failure means skip backend
            print(f"warning: could not create session for {provider}: {e}; skipping")
            continue
        actual = sess.get_providers()
        if provider not in actual:
            print(f"warning: requested {provider} but session runs on {actual} "
                  "(silent fallback?); timings may not isolate the backend")
        print(f"startup: backend={provider} session_providers={actual} "
              f"output={out_name} pooling={pooling}({pooling_src}) "
              f"normalize={normalize}({normalize_src})")
        stream = BatchStream(tokenized, args.batch_size, context_size, pack,
                             pad_id, texts, tok, shuffle, args.seed)
        try:
            stats = run_backend(sess, stream, feed_names, out_idx, pooling,
                                normalize, args.warmup_batches, tokens_limit,
                                minutes_limit, label, show_progress)
        except KeyboardInterrupt:
            print("\ninterrupted; printing partial results")
            interrupted = True
            stats = None
        if stats is not None:
            summary = summarize(stats)
            rows.append({"provider": provider, "label": label, "summary": summary})
            if args.verbose:
                print(f"detail: {label}: {summary['batches']} batches, "
                      f"{summary['throughput_overall_tps']:.0f} tok/s overall, "
                      f"{stats['epochs']} corpus epoch(s), "
                      f"feed=[{','.join(feed_names)}]")
        if interrupted:
            break
    if not rows:
        print("error: all backends failed; nothing to report", file=sys.stderr)
        return 2

    if args.output_format == "json":
        report = render_json(config_json, rows)
    elif args.output_format == "csv":
        report = render_csv(config_lines, rows)
    else:
        report = render_table(config_lines, rows)
    print(report)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"report written to {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
