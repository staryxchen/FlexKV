#!/usr/bin/env python3
"""Multi-layer layerwise benchmark: CE SDMA vs classic CTA kernel.

Simulates the real layerwise transfer scenario: many small per-layer transfers
back-to-back, where SDMA per-call launch overhead dominates.
"""
import argparse
import statistics
import time

import torch

from flexkv import c_ext


def bench_multi_layer(num_blocks, num_layers, iters, is_h2d, chunk_size, mode):
    kv_dim = 1  # MLA
    block_stride = chunk_size
    layer_stride = (num_blocks + 10) * block_stride
    kv_stride = chunk_size

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    gpu_tensors = [
        torch.empty(layer_stride * kv_dim // 8, dtype=torch.int64, device="cuda")
        for _ in range(num_layers)
    ]
    gpu_tensor_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_tensors], dtype=torch.int64).pin_memory()
    total_cpu_elems = layer_stride * kv_dim * num_layers // 8
    cpu_tensor = torch.empty(total_cpu_elems, dtype=torch.int64, pin_memory=True)

    use_ce = mode == "sdma"
    kwargs = dict(
        gpu_block_id_tensor=block_ids, gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
        gpu_kv_stride_in_bytes=kv_stride, gpu_block_stride_in_bytes=block_stride,
        gpu_layer_stride_in_bytes=layer_stride,
        cpu_block_id_tensor=block_ids, cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=kv_stride, cpu_layer_stride_in_bytes=layer_stride,
        cpu_block_stride_in_bytes=block_stride,
        chunk_size_in_bytes=chunk_size, start_layer_id=0, num_layers=num_layers,
        transfer_num_cta=4, is_host_to_device=is_h2d, use_ce_transfer=use_ce,
        is_mla=True, gpu_block_type=0, sync=True, ce_path_opt=True,
        ce_segment_threshold=8, ce_force_path=-1,
        ce_enable_memcpy2d=False, is_blockfirst=False)

    for _ in range(10):
        c_ext.transfer_kv_blocks(**kwargs)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c_ext.transfer_kv_blocks(**kwargs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times), min(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=100)
    args = p.parse_args()

    cases = [
        (4, 4, "4 blk x 4 lyr (4KB/lyr, 16KB total)"),
        (8, 8, "8 blk x 8 lyr (8KB/lyr, 64KB total)"),
        (4, 32, "4 blk x 32 lyr (4KB/lyr, 128KB total)"),
        (8, 32, "8 blk x 32 lyr (8KB/lyr, 256KB total)"),
        (16, 32, "16 blk x 32 lyr (16KB/lyr, 512KB total)"),
        (32, 32, "32 blk x 32 lyr (32KB/lyr, 1MB total)"),
        (4, 80, "4 blk x 80 lyr (4KB/lyr, 320KB total)"),
        (8, 80, "8 blk x 80 lyr (8KB/lyr, 640KB total)"),
    ]

    for mode in ["sdma", "classic"]:
        name = {"sdma": "CE_SDMA", "classic": "CLASSIC_KERNEL"}[mode]
        print(f"\n{'='*75}")
        print(f"  Mode: {name}")
        print(f"{'='*75}")
        print(f"{'Config':<50} {'H2D med':>10} {'D2H med':>10}")
        print("-" * 75)
        for nb, nl, label in cases:
            h2d_med, _ = bench_multi_layer(nb, nl, args.iters, True, 1024, mode)
            d2h_med, _ = bench_multi_layer(nb, nl, args.iters, False, 1024, mode)
            print(f"{label:<50} {h2d_med:>10.1f} {d2h_med:>10.1f}")
    print()


if __name__ == "__main__":
    main()
