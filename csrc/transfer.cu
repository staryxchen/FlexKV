/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ce_trace.h"
#include "monitoring/metrics_manager.h"
#include "transfer.cuh"
#include "ce_transfer.h"
#include "transfer_kernels.cuh"

namespace flexkv {

// ============================================================================
// Main host function
// ============================================================================

template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, void *cpu_ptr, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes,
    cudaStream_t stream, int transfer_num_cta, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla,
    int64_t gpu_block_stride_in_bytes, bool sync,
    const CETransferConfig &ce_config) {

  int64_t *cpu_ptr_int64 = reinterpret_cast<int64_t *>(cpu_ptr);
  int64_t cpu_kv_stride_int64 = cpu_kv_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_block_stride_int64 = cpu_block_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_layer_stride_int64 = cpu_layer_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / sizeof(int64_t);

  // CE transfer mode
  if (use_ce_transfer) {
    int kv_dim = is_mla ? 1 : 2;

    // Analyze block-id contiguity
    CEAnalysis analysis = analyze_ce_transfer(
        gpu_block_ids, cpu_block_ids, num_blocks,
        cpu_block_stride_in_bytes, chunk_size_in_bytes,
        gpu_block_stride_in_bytes);

    // path_opt_enabled off → PER_BLOCK; else choose_path() picks a strategy.
    if (!ce_config.path_opt_enabled) {
      ce_trace_log(static_cast<int>(Type), is_host_to_device,
                   num_blocks, start_layer_id, num_layers, is_mla, kv_dim,
                   chunk_size_in_bytes,
                   gpu_tensor_handler.gpu_kv_stride * (int64_t)sizeof(int64_t),
                   gpu_block_stride_in_bytes,
                   gpu_tensor_handler.gpu_layer_stride * (int64_t)sizeof(int64_t),
                   cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                   cpu_block_stride_in_bytes,
                   gpu_startoff_inside_chunks, cpu_startoff_inside_chunks,
                   ce_config, analysis, CEPath::PER_BLOCK,
                   gpu_block_ids, cpu_block_ids);
      ce_transfer_per_block<Type>(
          num_blocks, start_layer_id, num_layers, kv_dim,
          gpu_block_ids, gpu_tensor_handler,
          gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
          cpu_kv_stride_int64, cpu_layer_stride_int64,
          cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
          chunk_size_in_bytes, stream, is_host_to_device);
    } else {
      // force_path: benchmark only
      CEPath path;
      if (ce_config.force_path >= 0) {
        TORCH_CHECK(ce_config.force_path <= 4,
                    "force_path out of range [0,4]: ", ce_config.force_path);
        path = static_cast<CEPath>(ce_config.force_path);
      } else {
        // is_full_block: all layers*kv_dim in one call (rank0_only).
        // layer_parallel → !full_block; routes bfirst+MLA+D2H to S_SCT.
        bool is_full_block = ((int64_t)num_layers * kv_dim * chunk_size_in_bytes
                              == cpu_block_stride_in_bytes);
        path = choose_path(analysis, ce_config, chunk_size_in_bytes,
                           is_host_to_device, is_full_block);
      }

      ce_trace_log(static_cast<int>(Type), is_host_to_device,
                   num_blocks, start_layer_id, num_layers, is_mla, kv_dim,
                   chunk_size_in_bytes,
                   gpu_tensor_handler.gpu_kv_stride * (int64_t)sizeof(int64_t),
                   gpu_block_stride_in_bytes,
                   gpu_tensor_handler.gpu_layer_stride * (int64_t)sizeof(int64_t),
                   cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                   cpu_block_stride_in_bytes,
                   gpu_startoff_inside_chunks, cpu_startoff_inside_chunks,
                   ce_config, analysis, path,
                   gpu_block_ids, cpu_block_ids);

      switch (path) {
        case CEPath::CONTIG_DIRECT:
          ce_transfer_contig_direct<Type>(
              num_blocks, start_layer_id, num_layers, kv_dim,
              gpu_block_ids, gpu_tensor_handler,
              gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
              cpu_kv_stride_int64, cpu_layer_stride_int64,
              cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
              chunk_size_in_bytes, stream, is_host_to_device);
          break;
        case CEPath::SEGMENT_DIRECT:
          ce_transfer_segment_direct<Type>(
              num_blocks, start_layer_id, num_layers, kv_dim,
              gpu_block_ids, gpu_tensor_handler,
              gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
              cpu_kv_stride_int64, cpu_layer_stride_int64,
              cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
              chunk_size_in_bytes, stream, is_host_to_device, analysis,
              ce_config);
          break;
        case CEPath::SEGMENT_SCATTER:
          ce_transfer_segment_scatter<Type>(
              num_blocks, start_layer_id, num_layers, kv_dim,
              gpu_block_ids, gpu_tensor_handler,
              gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
              cpu_kv_stride_int64, cpu_layer_stride_int64,
              cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
              chunk_size_in_bytes, stream, is_host_to_device, analysis,
              ce_config);
          break;
        case CEPath::GATHER_SCATTER:
          ce_transfer_gather_scatter<Type>(
              num_blocks, start_layer_id, num_layers, kv_dim,
              gpu_block_ids, gpu_tensor_handler,
              gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
              cpu_kv_stride_int64, cpu_layer_stride_int64,
              cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
              chunk_size_in_bytes, stream, is_host_to_device, analysis,
              ce_config);
          break;
        case CEPath::GATHER_DIRECT:
          ce_transfer_gather_direct<Type>(
              num_blocks, start_layer_id, num_layers, kv_dim,
              gpu_block_ids, gpu_tensor_handler,
              gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64,
              cpu_kv_stride_int64, cpu_layer_stride_int64,
              cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
              chunk_size_in_bytes, stream, is_host_to_device, analysis,
              ce_config);
          break;
        default:
          break;
      }
    }  // end else (path_opt_enabled)
  } else {
    // Classic compute-kernel transfer (shared with ROCm).
    const int kv_dim = is_mla ? 1 : 2;
    CEAnalysis analysis = analyze_ce_transfer(
        gpu_block_ids, cpu_block_ids, num_blocks, cpu_block_stride_in_bytes,
        chunk_size_in_bytes, gpu_block_stride_in_bytes);
    ce_trace_log(static_cast<int>(Type), is_host_to_device, num_blocks,
                 start_layer_id, num_layers, is_mla, kv_dim,
                 chunk_size_in_bytes,
                 gpu_tensor_handler.gpu_kv_stride * (int64_t)sizeof(int64_t),
                 gpu_block_stride_in_bytes,
                 gpu_tensor_handler.gpu_layer_stride * (int64_t)sizeof(int64_t),
                 cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                 cpu_block_stride_in_bytes, gpu_startoff_inside_chunks,
                 cpu_startoff_inside_chunks, ce_config, analysis,
                 CEPath::CLASSIC_KERNEL, gpu_block_ids, cpu_block_ids);
    launch_classic_transfer_kv_blocks<Type>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks, chunk_size_in_bytes,
        stream, transfer_num_cta, is_mla, is_host_to_device);
  }
  if (sync) {
    cudaStreamSynchronize(stream);
  }
}

// Explicit template instantiations
template void transfer_kv_blocks<BackendType::VLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);

template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);

template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, int64_t, bool, const CETransferConfig &);

} // namespace flexkv
