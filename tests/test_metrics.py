import json

from onnx_benchy import __version__, footer
from onnx_benchy.metrics import fmt_latency, render_csv, render_json, render_table


def _rows():
    return [{
        "provider": "CPUExecutionProvider",
        "label": "cpu",
        "summary": {
            "batches": 10, "tokens": 1000, "elapsed_s": 1.0,
            "latency_ms": {"mean": 100.0, "std": 5.0},
            "throughput_tps": {"mean": 1000.0, "std": 50.0},
        },
    }]


def test_footer_matches_version():
    assert footer() == f"benchmark by onnx_benchy version {__version__}"


def test_latency_ms_vs_seconds():
    assert "ms" in fmt_latency(110.2, 4.1, 10)
    assert fmt_latency(1500.0, 100.0, 10).startswith("1.50 s")
    assert "n/a" in fmt_latency(10.0, 0.0, 1)


def test_table_ends_with_footer():
    text = render_table([("model", "m.onnx")], _rows())
    assert text.splitlines()[-1] == footer()
    assert "m.onnx" in text


def test_table_preserves_brackets_in_values():
    # rich markup would otherwise swallow [bracketed] path segments
    text = render_table([("model", "/tmp/[weird]/m.onnx")], _rows())
    assert "/tmp/[weird]/m.onnx" in text


def test_json_valid_and_has_generated_by():
    payload = json.loads(render_json({"batch_size": 32}, _rows()))
    assert payload["config"]["batch_size"] == 32
    assert payload["generated_by"] == footer()


def test_csv_ends_with_footer():
    text = render_csv([("model", "m.onnx")], _rows())
    assert text.splitlines()[-1] == footer()
    assert text.splitlines()[0].startswith("# model:")
