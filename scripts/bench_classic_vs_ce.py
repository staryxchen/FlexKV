#!/usr/bin/env python3
"""Classic CTA kernel vs CE paths (opt / PER_BLOCK)."""
from __future__ import annotations

import argparse
import time

import torch
from flexkv.c_ext import transfer_kv_blocks


WORKLOADS = [
    # (label, layers, blocks, chunk_bytes) — chunk ~ MLA page-ish or small layerwise
    ("small_lw  8x8x1K", 8, 8, 1024),
    ("med_lw   32x32x1K", 32, 32, 1024),
    ("large_lw 32x64x4K", 32, 64, 4096),
    ("bulk     61x128x16K", 61, 128, 16384),
]


def make_bufs(num_layers: int, num_blocks: int, chunk: int, scattered: bool):
    block_stride = chunk
    kv_stride = num_blocks * chunk
    layer_stride = 2 * kv_stride
    gpu_tensors = [
        torch.zeros(layer_stride // 8, dtype=torch.int64, device="cuda")
        for _ in range(num_layers)
    ]
    # Keep alive for data_ptr validity.
    keep = gpu_tensors
    gpu_tensor_ptrs = torch.tensor(
        [t.data_ptr() for t in gpu_tensors], dtype=torch.int64
    ).pin_memory()
    cpu_total = num_layers * layer_stride
    cpu_tensor = torch.arange(cpu_total // 8, dtype=torch.int64).pin_memory()
    if scattered:
        # Reverse order → many segments for CE path selection.
        ids = torch.arange(num_blocks - 1, -1, -1, dtype=torch.int64).pin_memory()
    else:
        ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    # clone() drops pin_memory; classic kernel reads host ids from device.
    cpu_ids = ids.detach().clone().pin_memory()
    base = dict(
        gpu_block_id_tensor=ids,
        gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
        gpu_kv_stride_in_bytes=kv_stride,
        gpu_block_stride_in_bytes=block_stride,
        gpu_layer_stride_in_bytes=layer_stride,
        cpu_block_id_tensor=cpu_ids,
        cpu_tensor=cpu_tensor,
        cpu_kv_stride_in_bytes=kv_stride,
        cpu_layer_stride_in_bytes=layer_stride,
        cpu_block_stride_in_bytes=block_stride,
        chunk_size_in_bytes=chunk,
        start_layer_id=0,
        num_layers=num_layers,
        is_mla=False,
        gpu_block_type=0,
        sync=True,
    )
    total_bytes = num_layers * 2 * num_blocks * chunk
    return base, total_bytes, keep


def time_call(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def configs(cta: int):
    return [
        ("CE_opt", dict(use_ce_transfer=True, ce_path_opt=True, ce_force_path=-1,
                        transfer_num_cta=cta)),
        ("CE_per_block", dict(use_ce_transfer=True, ce_path_opt=False, ce_force_path=-1,
                              transfer_num_cta=cta)),
        (f"classic_cta{cta}", dict(use_ce_transfer=False, transfer_num_cta=cta)),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--cta", type=int, default=16)
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()

    torch.cuda.set_device(args.device)
    print(f"device=cuda:{args.device}  cta={args.cta}  iters={args.iters}")
    print(f"{'workload':18s} {'pattern':8s} {'dir':4s} "
          f"{'CE_opt':>12s} {'CE_PB':>12s} "
          f"{'classic':>12s} {'cl/opt':>8s}")
    print("-" * 88)

    for label, layers, blocks, chunk in WORKLOADS:
        for scattered, pat in ((False, "contig"), (True, "scatter")):
            base, total_bytes, keep = make_bufs(layers, blocks, chunk, scattered)
            for is_h2d, direction in ((True, "H2D"), (False, "D2H")):
                results = {}
                for name, cfg in configs(args.cta):
                    def run(c=cfg, h2d=is_h2d):
                        transfer_kv_blocks(
                            **base, is_host_to_device=h2d, **c
                        )

                    ms = time_call(run, args.iters, args.warmup)
                    gbps = (total_bytes / 1e9) / (ms / 1e3)
                    results[name] = (ms, gbps)

                opt_ms = results["CE_opt"][0]
                classic_name = f"classic_cta{args.cta}"
                ratio = opt_ms / results[classic_name][0]

                def fmt(key):
                    ms, gbps = results[key]
                    return f"{ms:6.3f}ms/{gbps:4.1f}"

                print(
                    f"{label:18s} {pat:8s} {direction:4s} "
                    f"{fmt('CE_opt'):>12s} {fmt('CE_per_block'):>12s} "
                    f"{fmt(classic_name):>12s} "
                    f"{ratio:7.2f}x"
                )
            del keep


if __name__ == "__main__":
    main()
