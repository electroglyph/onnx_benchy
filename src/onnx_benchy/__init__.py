"""onnx_benchy: benchmark ONNX embedding models across ONNX Runtime backends."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("onnx_benchy")
except PackageNotFoundError:  # pragma: no cover - dev checkout without install
    __version__ = "0.0.0+unknown"

FOOTER_PREFIX = "benchmark by onnx_benchy version"


def footer() -> str:
    return f"{FOOTER_PREFIX} {__version__}"
