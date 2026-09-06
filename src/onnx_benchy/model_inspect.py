"""ONNX model inspection + pooling/normalization auto-detection.

Pooling info does NOT live in Transformers config.json or
config_sentence_transformers.json. It lives in the Sentence-Transformers
module layout:
  - modules.json: ordered module list; a ".../Pooling" entry means pooling
    is expected, a ".../Normalize" entry (dir like "2_Normalize/") means
    normalize=true.
  - <idx>_Pooling/config.json: the ONLY file carrying the pooling mode, in
    one of two schemas:
      verbose (ST <= 5.3): {"pooling_mode_mean_tokens": true, ...}
      compact (ST >= 5.4): {"pooling_mode": "mean", ...}
"""

from __future__ import annotations

import glob
import json
import os

import onnxruntime as ort

OUTPUT_PREFERENCE = ["sentence_embedding", "last_hidden_state", "token_embeddings"]

# Verbose-schema boolean key -> canonical pooling mode.
VERBOSE_POOLING_KEYS = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}

COMPACT_POOLING_MODES = {
    "cls",
    "max",
    "mean",
    "mean_sqrt_len_tokens",
    "weightedmean",
    "lasttoken",
}

# Modes benchy can actually execute.
IMPLEMENTED_POOLING = {"cls", "mean", "max", "lasttoken", "none"}


class ConfigError(Exception):
    """Raised for unresolvable/inconsistent model or pooling configuration."""


def inspect_model(model_path: str) -> dict:
    """Dry-run inspection: list model inputs/outputs via a CPU session."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return {
        "inputs": [
            {"name": i.name, "shape": list(i.shape), "dtype": str(i.type)}
            for i in sess.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "shape": list(o.shape), "dtype": str(o.type)}
            for o in sess.get_outputs()
        ],
    }


def format_outputs(info: dict) -> str:
    lines = ["model outputs:"]
    for idx, o in enumerate(info["outputs"]):
        lines.append(
            f"  [{idx}] name={o['name']} shape={o['shape']} dtype={o['dtype']}"
        )
    return "\n".join(lines)


def resolve_output(outputs: list[dict], requested: str) -> tuple[int, str]:
    """Resolve --output (auto | int index | exact name) to (index, name)."""
    names = [o["name"] for o in outputs]
    if requested == "auto":
        for preferred in OUTPUT_PREFERENCE:
            if preferred in names:
                return names.index(preferred), preferred
        return 0, names[0]
    try:
        idx = int(requested)
    except ValueError:
        idx = None
    if idx is not None:
        if -len(names) <= idx < len(names):
            return idx % len(names), names[idx % len(names)]
        raise ConfigError(
            f"--output index {requested} out of range; valid outputs:\n"
            + "\n".join(f"  [{i}] {n}" for i, n in enumerate(names))
        )
    if requested in names:
        return names.index(requested), requested
    hints = [n for n in names if requested.lower() in n.lower()]
    hint = f" Did you mean: {', '.join(hints)}?" if hints else ""
    raise ConfigError(
        f"--output {requested!r} not found in model outputs.{hint} Valid:\n"
        + "\n".join(f"  [{i}] {n}" for i, n in enumerate(names))
    )


def output_rank(outputs: list[dict], index: int) -> int | None:
    """Rank of the chosen output, or None if the shape itself is unknown."""
    shape = outputs[index]["shape"]
    # ORT reports dynamic dims as strings/None; rank is still len(shape).
    if shape is None:
        return None
    return len(shape)


def _read_json(path: str) -> dict | list | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, (dict, list)) else None
    except (OSError, json.JSONDecodeError):
        return None


def parse_pooling_config(cfg: dict) -> str | None:
    """Extract the pooling mode from a *Pooling/config.json dict.

    Supports both the verbose-boolean schema (ST <= 5.3) and the
    compact-string schema (ST >= 5.4). Returns None if no mode found.
    Raises ConfigError on multi-mode concatenation (dims would silently break).
    """
    if "pooling_mode" in cfg and isinstance(cfg["pooling_mode"], str):
        mode = cfg["pooling_mode"]
        if mode in COMPACT_POOLING_MODES:
            return mode
        return None
    active = [mode for key, mode in VERBOSE_POOLING_KEYS.items() if cfg.get(key) is True]
    if len(active) > 1:
        raise ConfigError(
            f"Pooling config enables multiple modes {active} (concatenated "
            "embeddings); pick one explicitly with --pooling cls|mean|max|lasttoken"
        )
    return active[0] if active else None


def find_st_configs(config_dir: str) -> tuple[dict | list | None, str | None, dict | None]:
    """Locate modules.json and the *Pooling/config.json under config_dir."""
    modules = None
    modules_path = os.path.join(config_dir, "modules.json")
    if os.path.isfile(modules_path):
        modules = _read_json(modules_path)
    pooling_cfg = None
    candidates = sorted(glob.glob(os.path.join(config_dir, "*Pooling", "config.json")))
    for path in candidates:
        cfg = _read_json(path)
        if isinstance(cfg, dict):  # skip corrupt/non-object configs, try next
            pooling_cfg = path
            pooling_parsed = cfg
            break
    else:
        pooling_parsed = None
    return modules, pooling_cfg, pooling_parsed


def _modules_has(modules: dict | list | None, kind: str) -> bool:
    if not modules:
        return False
    entries = modules.values() if isinstance(modules, dict) else modules
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_type = str(entry.get("type", ""))
        path = str(entry.get("path", ""))
        if kind in entry_type or (path and kind in path):
            return True
    return False


def detect_pooling_normalize(
    requested_pooling: str,
    requested_normalize: str,
    rank: int | None,
    config_dir: str,
) -> tuple[str, str, bool, str]:
    """Resolve effective (pooling, pooling_src, normalize, normalize_src).

    Priority: explicit flags > ONNX output rank > ST config dir > fallback.
    """
    # --- normalize source detection (independent of pooling) ---
    modules, pooling_cfg_path, pooling_cfg = find_st_configs(config_dir)
    st_normalize = _modules_has(modules, "Normalize") or bool(
        glob.glob(os.path.join(config_dir, "*Normalize"))
    )

    # --- pooling ---
    if requested_pooling == "auto":
        if rank == 2:
            pooling, pooling_src = "none", "onnx output is rank 2 (already pooled)"
        elif rank is not None and rank != 3:
            raise ConfigError(
                f"chosen output is rank {rank}; benchy needs a rank-2 (pooled) "
                "or rank-3 (per-token) embedding output"
            )
        else:
            mode = parse_pooling_config(pooling_cfg) if pooling_cfg else None
            if mode is not None:
                schema = (
                    "compact schema"
                    if "pooling_mode" in pooling_cfg
                    else "verbose schema"
                )
                if mode not in IMPLEMENTED_POOLING:
                    raise ConfigError(
                        f"model pooling config asks for {mode!r}; benchy cannot "
                        "execute it (silent mis-pooling risk). Re-run with "
                        "--pooling cls|mean|max|lasttoken"
                    )
                pooling = mode
                pooling_src = f"{os.path.basename(os.path.dirname(pooling_cfg_path))}/config.json, {schema}"
            elif rank == 3 or rank is None:
                pooling = "mean"
                pooling_src = (
                    "fallback guess (no ST pooling config found; override with --pooling)"
                    + ("; output rank unknown" if rank is None else "")
                )
    else:
        pooling = requested_pooling
        pooling_src = "explicit --pooling flag"
        if rank == 2 and pooling != "none":
            raise ConfigError(
                f"chosen output is rank 2 (already pooled) but --pooling {pooling}; "
                "use --pooling none"
            )
        if rank == 3 and pooling == "none":
            raise ConfigError(
                "chosen output is rank 3 (per-token) but --pooling none; "
                "pick --pooling cls|mean|max|lasttoken"
            )

    # --- normalize ---
    if requested_normalize == "auto":
        if st_normalize:
            normalize, normalize_src = True, "modules.json Normalize entry"
        else:
            normalize, normalize_src = (
                False,
                "default false (cosine-sim users usually want --normalize true)",
            )
    else:
        normalize = requested_normalize == "true"
        normalize_src = "explicit --normalize flag"

    return pooling, pooling_src, normalize, normalize_src


def build_feed_names(input_names: list[str]) -> list[str]:
    """Which model inputs benchy will feed (dynamic, from session inputs)."""
    if "input_ids" not in input_names:
        raise ConfigError(
            f"model inputs {input_names} contain no 'input_ids'; "
            "onnx_benchy only supports text-encoder embedding models"
        )
    feed = ["input_ids"]
    for optional in ("attention_mask", "token_type_ids"):
        if optional in input_names:
            feed.append(optional)
    return feed
