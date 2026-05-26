"""ROCm-specific dispatch smoke tests.

Complements ``test_backend_dispatch.py`` (which validates vendor-neutral
framework contracts) by verifying that on a ROCm machine the framework
actually routes to ``RocmBackend``, the ABC contract is fully satisfied
by the ROCm implementation, and ROCm-specific declarations
(visibility env vars, GDS=False, etc.) are correct.

These tests do NOT touch a real GPU — Stage 3 (``rocm_smoke.py``)
covers device / stream / pinned-host operations.

Skipped automatically when ``torch.version.hip`` is None.
"""
from __future__ import annotations

import importlib
import os

import pytest


def _is_rocm_env() -> bool:
    try:
        import torch
        return getattr(torch.version, "hip", None) is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_rocm_env(),
    reason="Requires a ROCm PyTorch install (torch.version.hip is set).",
)


# ---------------------------------------------------------------------------
# Routing — autodetection and explicit aliases
# ---------------------------------------------------------------------------

def test_current_backend_is_rocm(monkeypatch):
    """On a ROCm machine, autodetection must resolve to RocmBackend."""
    # Defensive: clear any forced override that earlier tests in the same
    # session might have set, then re-import the package.
    monkeypatch.delenv("FLEXKV_GPU_BACKEND", raising=False)
    import flexkv.gpu_backend as gb
    importlib.reload(gb)

    assert gb.current_backend.vendor is gb.GpuVendor.ROCM, (
        f"Expected ROCM, got {gb.current_backend.vendor}. "
        "Check _BUILTIN_BACKENDS detection order or RocmBackend.is_available()."
    )


def test_rocm_aliases_all_resolve_to_rocm_backend():
    """All three ROCm aliases documented in _FORCE_MAP must work."""
    from flexkv.gpu_backend import select_backend, GpuVendor
    for alias in ("rocm", "hip", "amd"):
        b = select_backend(alias)
        assert b.vendor is GpuVendor.ROCM, (
            f"alias '{alias}' resolved to {b.vendor}, expected ROCM"
        )


# ---------------------------------------------------------------------------
# ABC contract — does RocmBackend implement every @abstractmethod?
# ---------------------------------------------------------------------------

def test_rocm_backend_directly_instantiable():
    """If any @abstractmethod is missing, instantiation raises TypeError
    ('Can't instantiate abstract class RocmBackend with abstract methods xxx').
    A successful construction proves the ABC contract is fully satisfied."""
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    b = RocmBackend()
    assert b is not None


def test_rocm_backend_is_available():
    """RocmBackend.is_available() must be True on a ROCm PyTorch install.
    If False, autodetection would have fallen through to GenericBackend."""
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    assert RocmBackend.is_available() is True


# ---------------------------------------------------------------------------
# ROCm-specific declarations
# ---------------------------------------------------------------------------

def test_rocm_visible_devices_env_vars():
    """Per README_en.md §3.7, ROCm declares all three visibility masks
    with HIP_ as the canonical (write) target."""
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    expected = (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    )
    assert RocmBackend.visible_devices_env_vars() == expected


def test_rocm_torch_device_type_is_cuda():
    """ROCm tensors live on torch.device('cuda', N) — the same device-type
    string as NVIDIA. (MUSA is the odd one out, reporting 'musa'.)"""
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    assert RocmBackend.torch_device_type() == "cuda"


def test_rocm_no_direct_storage():
    """ROCm has no cuFile / GDS today; supports_direct_storage must be False
    so upper layers fall back to transfer_kv_blocks_ssd (POSIX/io_uring)."""
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    b = RocmBackend()
    assert b.supports_direct_storage() is False


# ---------------------------------------------------------------------------
# Python ↔ C++ binding wiring
# ---------------------------------------------------------------------------

def test_c_ext_module_importable():
    """Stage 1 produced flexkv.c_ext — verify the Python layer can load it."""
    import flexkv.c_ext as ext
    assert ext is not None


def test_c_ext_exposes_core_cross_vendor_symbols():
    """register_rocm_bindings(m) must register at minimum
    transfer_kv_blocks and TPTransferThreadGroup (README_en.md §3.2)."""
    import flexkv.c_ext as ext
    for name in ("transfer_kv_blocks", "TPTransferThreadGroup"):
        assert hasattr(ext, name), (
            f"flexkv.c_ext is missing '{name}'. "
            f"Check csrc/gpu_backend/rocm/rocm_bindings.cpp."
        )


def test_c_ext_does_not_expose_nvidia_only_symbols():
    """ROCm wheel must NOT expose GDS classes — they only exist under
    register_nvidia_bindings(m) gated by FLEXKV_ENABLE_GDS (README_en.md §3.5.1)."""
    import flexkv.c_ext as ext
    for name in ("GDSManager", "TPGDSTransferThreadGroup", "transfer_kv_blocks_gds"):
        assert not hasattr(ext, name), (
            f"flexkv.c_ext unexpectedly exposes '{name}' on ROCm. "
            f"GDS symbols must be NVIDIA-only."
        )
