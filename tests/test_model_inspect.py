import json

import pytest

from onnx_benchy.model_inspect import (
    ConfigError,
    detect_pooling_normalize,
    parse_pooling_config,
    resolve_output,
)

OUTPUTS = [
    {"name": "last_hidden_state", "shape": [1, "s", 384], "dtype": "tensor(float)"},
    {"name": "pooler_output", "shape": [1, 384], "dtype": "tensor(float)"},
]


def test_resolve_auto_prefers_pooled():
    outs = [
        {"name": "last_hidden_state", "shape": [], "dtype": ""},
        {"name": "sentence_embedding", "shape": [], "dtype": ""},
    ]
    assert resolve_output(outs, "auto") == (1, "sentence_embedding")


def test_resolve_auto_falls_back_to_index0():
    assert resolve_output(OUTPUTS, "auto") == (0, "last_hidden_state")


def test_resolve_int_and_name():
    assert resolve_output(OUTPUTS, "1") == (1, "pooler_output")
    assert resolve_output(OUTPUTS, "pooler_output") == (1, "pooler_output")
    with pytest.raises(ConfigError):
        resolve_output(OUTPUTS, "nope")
    with pytest.raises(ConfigError):
        resolve_output(OUTPUTS, "7")


def test_parse_verbose_schema():
    cfg = {"pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True,
           "pooling_mode_max_tokens": False}
    assert parse_pooling_config(cfg) == "mean"


def test_parse_compact_schema():
    assert parse_pooling_config({"pooling_mode": "cls"}) == "cls"


def test_parse_multi_mode_errors():
    with pytest.raises(ConfigError):
        parse_pooling_config({"pooling_mode_cls_token": True,
                              "pooling_mode_mean_tokens": True})


def test_detect_rank2_forces_none(tmp_path):
    pooling, _src, norm, _nsrc = detect_pooling_normalize("auto", "auto", 2, str(tmp_path))
    assert pooling == "none" and norm is False


def test_detect_from_st_dir_verbose(tmp_path):
    (tmp_path / "modules.json").write_text(json.dumps([
        {"idx": 0, "name": "0_Transformer", "path": "", "type": "transformers"},
        {"idx": 1, "name": "1_Pooling", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2_Normalize", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
    ]))
    pool_dir = tmp_path / "1_Pooling"
    pool_dir.mkdir()
    (pool_dir / "config.json").write_text(json.dumps(
        {"pooling_mode_mean_tokens": True, "pooling_mode_cls_token": False}))
    pooling, src, norm, _nsrc = detect_pooling_normalize("auto", "auto", 3, str(tmp_path))
    assert pooling == "mean" and "verbose" in src
    assert norm is True


def test_detect_explicit_wins_and_rank_checks(tmp_path):
    pooling, _, _, _ = detect_pooling_normalize("cls", "true", 3, str(tmp_path))
    assert pooling == "cls"
    with pytest.raises(ConfigError):  # rank2 + mean is contradictory
        detect_pooling_normalize("mean", "auto", 2, str(tmp_path))
    with pytest.raises(ConfigError):  # rank3 + none is contradictory
        detect_pooling_normalize("none", "auto", 3, str(tmp_path))


def test_non_dict_pooling_config_skipped(tmp_path):
    pool_dir = tmp_path / "1_Pooling"
    pool_dir.mkdir()
    (pool_dir / "config.json").write_text("[1, 2, 3]")  # valid JSON, not an object
    pooling, src, _, _ = detect_pooling_normalize("auto", "auto", 3, str(tmp_path))
    assert pooling == "mean" and "fallback" in src


def test_unexpected_rank_rejected(tmp_path):
    with pytest.raises(ConfigError):
        detect_pooling_normalize("auto", "auto", 4, str(tmp_path))
    with pytest.raises(ConfigError):
        detect_pooling_normalize("auto", "auto", 1, str(tmp_path))
