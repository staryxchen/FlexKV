import pytest
import torch


pytestmark = pytest.mark.skipif(
    not bool(getattr(torch.version, "hip", None)), reason="requires ROCm PyTorch"
)


def test_rocm_backend_defaults_to_ce():
    from flexkv import c_ext
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    # Product default remains CE (SDMA); classic kernel is opt-in via env=0.
    assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d
    assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h
    assert not GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d
    assert not hasattr(c_ext, "GDSManager")
    assert not hasattr(c_ext, "ANSTransferContext")


def test_rocm_host_runtime_selection():
    from flexkv.transfer import host_buffer

    assert host_buffer._is_rocm()


def test_rocm_classic_kernel_path_accepted():
    """use_ce_transfer=False must not raise; it launches CLASSIC_KERNEL."""
    from flexkv.c_ext import transfer_kv_blocks

    num_layers = 1
    num_blocks = 2
    chunk_size = 1024  # 16B-aligned → float4 classic path
    # VLLM: one ptr per layer; K/V packed via kv_stride.
    gpu_block_stride = chunk_size
    gpu_kv_stride = num_blocks * chunk_size
    gpu_layer_stride = 2 * gpu_kv_stride
    cpu_block_stride = chunk_size
    cpu_kv_stride = num_blocks * chunk_size
    cpu_layer_stride = 2 * cpu_kv_stride

    # Classic kernel reads host metadata from device; all host tensors must be pinned.
    gpu_tensors = [
        torch.zeros(gpu_layer_stride // 8, dtype=torch.int64, device="cuda")
        for _ in range(num_layers)
    ]
    gpu_tensor_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_tensors], dtype=torch.int64
    ).pin_memory()
    cpu_elems = cpu_layer_stride // 8
    cpu_tensor = torch.arange(cpu_elems, dtype=torch.int64).pin_memory()
    gpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()

    transfer_kv_blocks(
        gpu_block_id_tensor=gpu_block_ids,
        gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
        gpu_kv_stride_in_bytes=gpu_kv_stride,
        gpu_block_stride_in_bytes=gpu_block_stride,
        gpu_layer_stride_in_bytes=gpu_layer_stride,
        cpu_block_id_tensor=cpu_block_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=cpu_kv_stride,
        cpu_layer_stride_in_bytes=cpu_layer_stride,
        cpu_block_stride_in_bytes=cpu_block_stride,
        chunk_size_in_bytes=chunk_size,
        start_layer_id=0,
        num_layers=num_layers,
        transfer_num_cta=1,
        is_host_to_device=True,
        use_ce_transfer=False,
        is_mla=False,
        gpu_block_type=0,
        sync=True,
    )
    torch.cuda.synchronize()

    for block in range(num_blocks):
        for kv in range(2):
            cpu_off = (kv * cpu_kv_stride + block * cpu_block_stride) // 8
            gpu_off = (kv * gpu_kv_stride + block * gpu_block_stride) // 8
            n = chunk_size // 8
            assert torch.equal(
                gpu_tensors[0][gpu_off : gpu_off + n].cpu(),
                cpu_tensor[cpu_off : cpu_off + n],
            )
