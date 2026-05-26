"""ROCm runtime smoke tests.

These tests REQUIRE a real ROCm GPU. They exercise the GpuBackend
methods in groups 2-7 of docs/gpu_backend/README_en.md §3.1, using the
current backend resolved at module import time.

Layout: each test pins ONE capability so a failure points at exactly
which group is broken (devices / streams / pinned host / IPC / ...).
Stage 2 tests (test_rocm_dispatch.py) cover dispatch / ABC / declarations
without touching a real GPU; this file is the runtime counterpart.

Skipped automatically on non-ROCm or zero-device environments.
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Module-level gate: must be ROCm AND have at least one visible GPU.
# ---------------------------------------------------------------------------

def _is_rocm_with_gpu() -> bool:
    try:
        if getattr(torch.version, "hip", None) is None:
            return False
        from flexkv.gpu_backend import current_backend, GpuVendor
        if current_backend.vendor is not GpuVendor.ROCM:
            return False
        return current_backend.device_count() > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_rocm_with_gpu(),
    reason="Requires a ROCm machine with at least one visible GPU.",
)


@pytest.fixture(scope="module")
def backend():
    from flexkv.gpu_backend import current_backend
    return current_backend


# ===========================================================================
# Group 2: Devices
# ===========================================================================

def test_device_count_positive(backend):
    n = backend.device_count()
    assert n > 0, f"device_count() = {n}"


def test_device_name_nonempty(backend):
    name = backend.device_name()
    assert isinstance(name, str) and name


def test_set_and_current_device(backend):
    backend.set_device(0)
    cur = backend.current_device()
    assert cur == 0, f"current_device() after set_device(0) = {cur}"


def test_get_device_capability(backend):
    cap = backend.get_device_capability(0)
    assert isinstance(cap, tuple) and len(cap) == 2


def test_make_device_returns_cuda_typed(backend):
    """ROCm tensors live on torch.device('cuda', N) — same as NVIDIA."""
    dev = backend.make_device(0)
    assert dev.type == "cuda" and dev.index == 0


def test_synchronize_does_not_raise(backend):
    backend.synchronize()


def test_empty_cache_does_not_raise(backend):
    backend.empty_cache()


def test_init_runtime_is_idempotent(backend):
    backend.init_runtime()
    assert backend.is_initialized() is True
    backend.init_runtime()  # second call must be a no-op


def test_is_gpu_tensor_distinguishes_cpu_and_gpu(backend):
    cpu_t = torch.empty(4)
    assert backend.is_gpu_tensor(cpu_t) is False
    gpu_t = cpu_t.to(backend.make_device(0))
    assert backend.is_gpu_tensor(gpu_t) is True


# ===========================================================================
# Group 3: Streams
# ===========================================================================

def test_create_destroy_stream(backend):
    s = backend.create_stream(0)
    try:
        h = backend.stream_handle(s)
        assert isinstance(h, int) and h > 0, f"stream_handle = {h!r}"
    finally:
        backend.destroy_stream(s)


def test_current_stream_handle_is_int(backend):
    s = backend.get_current_stream()
    h = backend.stream_handle(s)
    assert isinstance(h, int)


def test_stream_context_manager(backend):
    s = backend.create_stream(0)
    try:
        with backend.stream_context(s):
            pass  # entering/exiting must not raise
    finally:
        backend.destroy_stream(s)


# ===========================================================================
# Group 4: Pinned host memory
# Critical: this is the only path that dlopens libamdhip64.so from Python.
#
# Note: alloc_pinned / free_pinned are *optional* in the ABC (interface.py
# leaves them as default-NotImplementedError). FlexKV upper layers go
# exclusively through register_host_tensor (page-lock an existing tensor),
# never through alloc_pinned (allocate from scratch). NVIDIA backend
# doesn't implement them either. So we only test the required pair.
# ===========================================================================

def test_register_unregister_host_tensor(backend):
    t = torch.empty(1024, dtype=torch.float16)
    backend.register_host_tensor(t)
    backend.unregister_host_tensor(t)


# ===========================================================================
# Group 6: IPC (capability-gated)
# Per-test gate via backend.supports_ipc() — some container environments
# disable IPC namespaces, in which case the backend reports False and we
# skip rather than fail.
# ===========================================================================

def test_supports_ipc_returns_bool(backend):
    out = backend.supports_ipc()
    assert isinstance(out, bool)


def test_ipc_handle_export(backend):
    """Verify that get_ipc_handle exports a non-empty byte string from a
    GPU tensor.

    We deliberately do NOT round-trip through open_ipc_handle in the same
    process. CUDA / HIP both specify that *IpcOpenMemHandle is callable
    *only by the importing process* — i.e. a process different from the
    exporter. ROCm 7.x enforces this strictly (returns error 201);
    NVIDIA's older runtimes were more permissive, but that's vendor
    happenstance, not a contract. Round-tripping in the same process
    here would test something that is *designed not to work*.

    True cross-process round-trip is exercised by Stage 4's
    test_memory_handle.py, which spawns a child process and passes the
    handle through it (the actual server-client topology).

    The handle is opaque bytes (64B on both CUDA and HIP per
    README_en.md §3.6.3, but we don't hard-assert the size — only that
    it's non-empty and a bytes-like object).
    """
    if not backend.supports_ipc():
        pytest.skip("Backend reports supports_ipc()=False.")

    src = torch.arange(16, dtype=torch.float32, device=backend.make_device(0))
    handle = backend.get_ipc_handle(src)
    assert isinstance(handle, (bytes, bytearray)), type(handle)
    assert len(handle) > 0, f"handle is empty: {handle!r}"


# ===========================================================================
# Group 7: Direct storage — already declarative-tested in Stage 2
# (test_rocm_no_direct_storage), no runtime test needed here.
# ===========================================================================
