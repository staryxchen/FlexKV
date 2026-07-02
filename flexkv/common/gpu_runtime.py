"""GPU runtime (CUDA / ROCm) ctypes adaptation layer.

FlexKV drives a small number of low-level GPU runtime calls (IPC memory
handles, pinned host memory registration) directly via ``ctypes`` instead of
going through PyTorch, because PyTorch does not expose them. This module
centralizes the platform detection and the CUDA-runtime <-> HIP-runtime
symbol-name mapping needed to make those calls work unchanged on both
NVIDIA (CUDA) and AMD (ROCm) GPUs.

On ROCm, PyTorch is built against HIP: ``torch.version.hip`` is set (and
``torch.version.cuda`` is ``None``). The HIP runtime library is
``libamdhip64.so`` and its runtime API uses a ``hip*`` prefix instead of
``cuda*`` (e.g. ``hipIpcGetMemHandle`` instead of ``cudaIpcGetMemHandle``).
The underlying struct layouts and integer constants are identical between
CUDA and HIP for the APIs used here.
"""

import ctypes
from typing import Optional

import torch

# True on ROCm builds of PyTorch (torch.version.hip is non-None), False on
# native CUDA builds.
IS_ROCM: bool = torch.version.hip is not None

# Function-name prefix used by the active GPU runtime.
_PREFIX = "hip" if IS_ROCM else "cuda"

# Mapping from a short logical name to the actual runtime symbol name.
# Extend this dict if more runtime functions need to be called via ctypes.
RUNTIME_FN = {
    "IpcGetMemHandle": _PREFIX + "IpcGetMemHandle",
    "IpcOpenMemHandle": _PREFIX + "IpcOpenMemHandle",
    "HostRegister": _PREFIX + "HostRegister",
    "HostUnregister": _PREFIX + "HostUnregister",
    "HostAlloc": _PREFIX + "HostAlloc",
    "FreeHost": _PREFIX + "FreeHost",
}

# IPC memory handle size in bytes. Both cudaIpcMemHandle_t and
# hipIpcMemHandle_t are 64-byte opaque structs on Linux x86_64.
IPC_HANDLE_SIZE = 64

# cudaIpcMemLazyEnablePeerAccess == hipIpcMemLazyEnablePeerAccess == 1
IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1

# cudaHostRegisterPortable/Mapped == hipHostRegisterPortable/Mapped
HOST_REGISTER_PORTABLE = 0x01
HOST_REGISTER_MAPPED = 0x02

# cudaSuccess == hipSuccess == 0
SUCCESS = 0


def load_gpu_runtime() -> ctypes.CDLL:
    """Load the active GPU runtime shared library (HIP on ROCm, CUDA
    otherwise) via ctypes."""
    if IS_ROCM:
        try:
            return ctypes.CDLL("libamdhip64.so")
        except OSError as e:
            raise RuntimeError(f"libamdhip64.so is unavailable: {e}") from e

    last_error: Optional[OSError] = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_error = e
    raise RuntimeError(f"libcudart.so is unavailable: {last_error}")


def get_runtime_func(runtime: ctypes.CDLL, logical_name: str):
    """Look up a runtime function by its logical (CUDA-style) name, e.g.
    ``get_runtime_func(rt, "IpcGetMemHandle")`` resolves to
    ``rt.cudaIpcGetMemHandle`` on CUDA or ``rt.hipIpcGetMemHandle`` on ROCm.
    """
    return getattr(runtime, RUNTIME_FN[logical_name])
