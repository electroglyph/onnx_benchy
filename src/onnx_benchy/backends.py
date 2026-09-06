"""Backend (execution provider) discovery and flag mapping.

Only the mainstream providers in SUPPORTED_PROVIDERS are benchmarkable.
Anything else the ORT build reports (Azure, ACL, ArmNN, NNAPI, TVM, CANN,
VitisAI, Rockchip, XNNPACK, WebGPU, ...) is always ignored — never
benchmarked, never selectable via a flag.
"""

from __future__ import annotations

# Canonical order: accelerators first, CPU last.
SUPPORTED_PROVIDERS: list[str] = [
    "CUDAExecutionProvider",
    "TensorrtExecutionProvider",  # note lowercase "rt"
    "ROCmExecutionProvider",
    "MIGraphXExecutionProvider",
    "OpenVINOExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",  # NOT "DirectMLExecutionProvider"
    "QNNExecutionProvider",
    "CPUExecutionProvider",
]

FLAG_TO_PROVIDER: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "rocm": "ROCmExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "directml": "DmlExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

# Short display names used in the results table.
PROVIDER_SHORT: dict[str, str] = {
    "CUDAExecutionProvider": "cuda",
    "TensorrtExecutionProvider": "tensorrt",
    "ROCmExecutionProvider": "rocm",
    "MIGraphXExecutionProvider": "migraphx",
    "OpenVINOExecutionProvider": "openvino",
    "CoreMLExecutionProvider": "coreml",
    "DmlExecutionProvider": "directml",
    "QNNExecutionProvider": "qnn",
    "CPUExecutionProvider": "cpu",
}


def resolve_backends(
    requested_flags: list[str], available: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Resolve which providers to benchmark.

    Args:
        requested_flags: CLI flag names given by the user (e.g. ["cuda"]),
            empty (or ["all"]) means auto mode.
        available: providers from ort.get_available_providers().

    Returns:
        (resolved, warnings, ignored) where resolved is the ordered list of
        provider strings to benchmark, warnings are skip messages for
        requested-but-unavailable providers, and ignored are available-but-
        unsupported providers that were skipped.
    """
    ignored = [p for p in available if p not in SUPPORTED_PROVIDERS]
    flags = [f for f in requested_flags if f != "all"]
    if not flags:
        # Auto mode: every SUPPORTED provider that is available, canonical order.
        resolved = [p for p in SUPPORTED_PROVIDERS if p in available]
        return resolved, [], ignored
    resolved, warnings = [], []
    for flag in flags:
        provider = FLAG_TO_PROVIDER[flag]
        if provider in available:
            if provider not in resolved:
                resolved.append(provider)
        else:
            warnings.append(
                f"skipped --{flag} ({provider} not available)"
            )
    return resolved, warnings, ignored
