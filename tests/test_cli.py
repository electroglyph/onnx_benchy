from onnx_benchy.backends import FLAG_TO_PROVIDER, resolve_backends


def test_auto_mode_intersects_supported():
    available = ["AzureExecutionProvider", "CPUExecutionProvider", "CUDAExecutionProvider"]
    resolved, warnings, ignored = resolve_backends([], available)
    assert resolved == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ignored == ["AzureExecutionProvider"]
    assert warnings == []


def test_explicit_missing_warns():
    resolved, warnings, _ = resolve_backends(["cuda", "cpu"], ["CPUExecutionProvider"])
    assert resolved == ["CPUExecutionProvider"]
    assert any("--cuda" in w for w in warnings)


def test_all_flags_map_to_nine_providers():
    assert len(FLAG_TO_PROVIDER) == 9
    assert set(FLAG_TO_PROVIDER) == {
        "cpu", "cuda", "tensorrt", "rocm", "migraphx",
        "openvino", "coreml", "directml", "qnn",
    }


def test_list_backends_exits_zero(capsys):
    from onnx_benchy.cli import main
    assert main(["--list-backends"]) == 0
    assert "CPUExecutionProvider" in capsys.readouterr().out


def test_missing_tokenizer_is_error(capsys):
    from onnx_benchy.cli import main
    assert main(["model.onnx"]) == 2


def test_bad_config_dir_is_error(tmp_path, capsys):
    from onnx_benchy.cli import main
    model = tmp_path / "m.onnx"
    model.write_bytes(b"fake")
    assert main([str(model), "--tokenizer", "x",
                 "--config-dir", str(tmp_path / "nope")]) == 2
