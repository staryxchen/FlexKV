/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Classic KV-block transfer kernels shared by CUDA (transfer.cu) and ROCm
 * (ce_transfer_dispatch.cu) for use_ce_transfer=false.
 *
 * CUDA builds use PTX ld.global.nc / st.global.cg so D2H can write pinned host
 * memory. ROCm builds use __builtin_nontemporal_load/store for the same intent.
 * Loop structure, CTA shape (transfer_num_cta x 1024), and float4 vs 8b
 * selection match the historical CUDA classic path.
 */
#pragma once

#include "gtensor_handler.cuh"
#include "rocm_utils.h"

namespace flexkv {

#define FLEXKV_FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

constexpr int kClassicFloat4AlignBytes = 16;

inline bool use_float4_kernel_path(int64_t chunk_size_in_bytes,
                                   int64_t gpu_startoff_inside_chunks,
                                   int64_t cpu_startoff_inside_chunks) {
  return (chunk_size_in_bytes % kClassicFloat4AlignBytes == 0) &&
         (gpu_startoff_inside_chunks % kClassicFloat4AlignBytes == 0) &&
         (cpu_startoff_inside_chunks % kClassicFloat4AlignBytes == 0);
}

// ---- Device load/store with host-pinned-safe semantics ----

__device__ __forceinline__ int64_t classic_load_u64(const int64_t *addr) {
#if defined(FLEXKV_USE_ROCM)
  return __builtin_nontemporal_load(addr);
#else
  int64_t element;
  asm volatile("ld.global.nc.u64 %0, [%1];"
               : "=l"(element)
               : "l"(addr)
               : "memory");
  return element;
#endif
}

__device__ __forceinline__ void classic_store_u64(int64_t *addr, int64_t value) {
#if defined(FLEXKV_USE_ROCM)
  __builtin_nontemporal_store(value, addr);
#else
  asm volatile("st.global.cg.u64 [%0], %1;" ::"l"(addr), "l"(value) : "memory");
#endif
}

__device__ __forceinline__ float4 classic_load_f4(const float4 *addr) {
#if defined(FLEXKV_USE_ROCM)
  float4 element;
  const float *p = reinterpret_cast<const float *>(addr);
  element.x = __builtin_nontemporal_load(p + 0);
  element.y = __builtin_nontemporal_load(p + 1);
  element.z = __builtin_nontemporal_load(p + 2);
  element.w = __builtin_nontemporal_load(p + 3);
  return element;
#else
  float4 element;
  asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];"
               : "=f"(element.x), "=f"(element.y), "=f"(element.z),
                 "=f"(element.w)
               : "l"(addr)
               : "memory");
  return element;
#endif
}

__device__ __forceinline__ void classic_store_f4(float4 *addr, float4 value) {
#if defined(FLEXKV_USE_ROCM)
  float *p = reinterpret_cast<float *>(addr);
  __builtin_nontemporal_store(value.x, p + 0);
  __builtin_nontemporal_store(value.y, p + 1);
  __builtin_nontemporal_store(value.z, p + 2);
  __builtin_nontemporal_store(value.w, p + 3);
#else
  asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" ::"l"(addr),
               "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
               : "memory");
#endif
}

// 8-byte (int64) copy path for MLA-sharded D2H where per-TP shard offsets are
// 8-aligned but not 16-aligned (e.g. DSv4 bytes_per_page_padded=37440, TP=8).
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel_8b(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    int64_t *cpu_chunk_ptr =
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks;

    int64_t *gpu_ptr =
        ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    int64_t *gpu_chunk_ptr =
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

    int64_t *src_chunk_ptr = is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
    int64_t *dst_chunk_ptr = is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

    for (int64_t idx = lane_id; idx < copy_size; idx += 32) {
      int64_t element = classic_load_u64(&src_chunk_ptr[idx]);
      classic_store_u64(&dst_chunk_ptr[idx], element);
    }
  }
}

// float4 (16B) classic path — backend type determined at compile time.
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  // Fold kv_dim into an inner loop so each warp iteration processes all KV
  // slots of one (layer, block). For non-MLA this halves the outer iteration
  // count, amortizing setup cost over 2x the useful work per warp step.
  int num_chunks = num_layers * num_blocks;
  int64_t copy_size_in_float4 = copy_size * sizeof(int64_t) / sizeof(float4);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / num_blocks;
    int block_pos = chunk_idx % num_blocks;
    int gpu_block_idx = gpu_block_ids[block_pos];
    int cpu_block_idx = cpu_block_ids[block_pos];

    int64_t *cpu_base = cpu_ptr + layer_idx * cpu_layer_stride +
                        cpu_block_idx * cpu_block_stride +
                        cpu_startoff_inside_chunks;

#pragma unroll
    for (int kv_idx = 0; kv_idx < kv_dim; kv_idx++) {
      int64_t *cpu_chunk_ptr = cpu_base + kv_idx * cpu_kv_stride;

      int64_t *gpu_ptr =
          ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
      int64_t *gpu_chunk_ptr =
          reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

      int64_t *src_chunk_ptr =
          is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
      int64_t *dst_chunk_ptr =
          is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

      for (int64_t idx = lane_id; idx < copy_size_in_float4; idx += 32) {
        float4 element =
            classic_load_f4(&FLEXKV_FLOAT4_PTR(src_chunk_ptr)[idx]);
        classic_store_f4(&FLEXKV_FLOAT4_PTR(dst_chunk_ptr)[idx], element);
      }
    }
  }
}

// Host launcher: CTA model matches CUDA classic (1024 threads x transfer_num_cta).
// gpu/cpu startoff arguments are in bytes (same as transfer_kv_blocks).
template <BackendType Type>
inline void launch_classic_transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64, int64_t cpu_startoff_inside_chunks,
    int64_t chunk_size_in_bytes, cudaStream_t stream, int transfer_num_cta,
    bool is_mla, bool is_host_to_device) {
  const int block_size = 1024;
  const int block_count = transfer_num_cta;
  const int64_t chunk_size_in_int64 =
      chunk_size_in_bytes / static_cast<int64_t>(sizeof(int64_t));
  const int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / static_cast<int64_t>(sizeof(int64_t));
  const int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / static_cast<int64_t>(sizeof(int64_t));

  dim3 blockDim(block_size);
  dim3 gridDim(block_count);

  // Choose the float4 (16B) vs int64 (8B) copy path based on alignment; the
  // 8b path handles MLA-sharded D2H where per-TP shard offsets are 8-aligned
  // but not 16-aligned (e.g. DSv4).
  const bool float4_path = use_float4_kernel_path(
      chunk_size_in_bytes, gpu_startoff_inside_chunks,
      cpu_startoff_inside_chunks);

  if (float4_path) {
    transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_int64, is_mla, is_host_to_device);
  } else {
    transfer_kv_blocks_kernel_8b<Type><<<gridDim, blockDim, 0, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_int64, is_mla, is_host_to_device);
  }
}

} // namespace flexkv
