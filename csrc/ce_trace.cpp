/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Implementation of CE transfer tracing. See ce_trace.h for the public API.
 *
 * Design:
 *   - spdlog async_logger with dedicated thread pool: calling thread only
 *     formats the JSON and enqueues; file I/O happens on the logger's
 *     background thread.  Per-call overhead on the hot path is ~2-5 us
 *     (JSON build + queue push), no open/flock/write/close per entry.
 *   - rotating_file_sink_mt: automatic file rotation by size + count.
 *   - std::atomic<bool> g_enabled: zero-overhead when disabled (single
 *     relaxed atomic load).
 *   - JSON built with fmt::format (bundled with spdlog).
 *   - Block IDs truncated to FLEXKV_CE_TRACE_MAX_BLOCKS (default 256) for
 *     human readability; 0 = full dump for complete replay.
 */
#include "ce_trace.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

namespace flexkv {

namespace {

// ---------------------------------------------------------------------------
// Cached configuration (initialized once on first access)
// ---------------------------------------------------------------------------

std::atomic<bool> g_enabled{false};
std::atomic<bool> g_initialized{false};
std::string g_file_path;
int g_max_blocks = 256;
std::mutex g_init_mutex;
std::atomic<uint64_t> g_trace_counter{0};

const char *kBackendNames[] = {"VLLM", "TRTLLM", "SGLANG"};

/// spdlog async logger — initialized lazily on first ce_trace_log().
std::shared_ptr<spdlog::logger> g_logger;

/// Initialize config + spdlog logger from env vars.  Runs at most once.
void ensure_initialized() {
  if (g_initialized.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized.load(std::memory_order_relaxed)) {
    return;
  }

  // ---- env vars ----
  const char *env = std::getenv("FLEXKV_CE_TRACE");
  bool enabled = false;
  if (env) {
    std::string val(env);
    enabled = (val == "1" || val == "true" || val == "yes" ||
               val == "TRUE" || val == "YES");
  }
  g_enabled.store(enabled, std::memory_order_relaxed);

  env = std::getenv("FLEXKV_CE_TRACE_FILE");
  g_file_path = (env && env[0] != '\0')
                    ? std::string(env)
                    : "./flexkv_ce_trace.jsonl";

  env = std::getenv("FLEXKV_CE_TRACE_MAX_BLOCKS");
  if (env) {
    int mb = std::atoi(env);
    g_max_blocks = (mb < 0) ? 0 : mb;
  }

  // ---- spdlog async logger setup ----
  // Only set up the logger if tracing is enabled, to avoid creating a
  // background thread when the feature is off.
  if (enabled) {
    // Initialize the async thread pool (queue size 8192, 1 worker thread).
    // The pool is shared across all loggers; init must happen before logger
    // creation.
    spdlog::init_thread_pool(8192, 1);

    // Rotating file sink: 50 MB per file, keep 5 files.
    auto sink = std::make_shared<
        spdlog::sinks::rotating_file_sink_mt>(g_file_path, 50 * 1024 * 1024, 5);

    auto logger = std::make_shared<spdlog::async_logger>(
        "ce_trace", sink, spdlog::thread_pool(),
        spdlog::async_overflow_policy::block);

    // We build our own JSON, so disable spdlog's formatter to avoid double
    // formatting.  Use a pattern that just passes the message through.
    logger->set_pattern("%v");
    logger->set_level(spdlog::level::info);
    logger->flush_on(spdlog::level::info);  // flush after every entry

    spdlog::register_logger(logger);
    g_logger = logger;
  }

  g_initialized.store(true, std::memory_order_release);
}

/// Append a JSON-escaped int64 array to the string, with optional truncation.
/// Returns true if the array was truncated.
bool append_int64_array(std::string &out, const char *field_name,
                        const int64_t *ids, int num_blocks, int max_blocks) {
  bool truncated = (max_blocks > 0 && num_blocks > max_blocks);
  int log_count = truncated ? max_blocks : num_blocks;

  out += field_name;
  out += ":[";
  char buf[32];
  for (int i = 0; i < log_count; ++i) {
    if (i > 0) out += ',';
    snprintf(buf, sizeof(buf), "%lld", (long long)ids[i]);
    out += buf;
  }
  out += "]";
  return truncated;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool ce_trace_enabled() {
  if (!g_initialized.load(std::memory_order_acquire)) {
    ensure_initialized();
  }
  return g_enabled.load(std::memory_order_relaxed);
}

void ce_trace_set_enabled(bool enabled) {
  ensure_initialized();
  g_enabled.store(enabled, std::memory_order_relaxed);
  // If enabling at runtime and logger wasn't created (because env var was
  // off at init time), force re-initialization by resetting the flag.
  if (enabled && !g_logger) {
    g_initialized.store(false, std::memory_order_release);
    ensure_initialized();
  }
}

void ce_trace_shutdown() {
  if (g_logger) {
    g_logger->flush();
    spdlog::drop("ce_trace");
    g_logger.reset();
  }
  spdlog::shutdown();
}

const std::string &ce_trace_file_path() {
  ensure_initialized();
  return g_file_path;
}

int ce_trace_max_blocks() {
  ensure_initialized();
  return g_max_blocks;
}

const char *ce_path_name(CEPath path) {
  switch (path) {
    case CEPath::PER_BLOCK:       return "PER_BLOCK";
    case CEPath::CONTIG_DIRECT:   return "CONTIG_DIRECT";
    case CEPath::SEGMENT_DIRECT:  return "SEGMENT_DIRECT";
    case CEPath::SEGMENT_SCATTER: return "SEGMENT_SCATTER";
    case CEPath::GATHER_SCATTER:  return "GATHER_SCATTER";
    case CEPath::GATHER_DIRECT:   return "GATHER_DIRECT";
    case CEPath::CLASSIC_KERNEL:  return "CLASSIC_KERNEL";
    default:                      return "UNKNOWN";
  }
}

const char *backend_type_name(int backend_type) {
  if (backend_type >= 0 && backend_type <= 2) {
    return kBackendNames[backend_type];
  }
  return "UNKNOWN";
}

void ce_trace_log(
    int backend_type,
    bool is_host_to_device,
    int num_blocks,
    int start_layer_id,
    int num_layers,
    bool is_mla,
    int kv_dim,
    int64_t chunk_size_in_bytes,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t gpu_startoff_inside_chunks,
    int64_t cpu_startoff_inside_chunks,
    const CETransferConfig &ce_config,
    const CEAnalysis &ce_analysis,
    CEPath ce_path,
    const int64_t *gpu_block_ids,
    const int64_t *cpu_block_ids) {

  ensure_initialized();
  if (!g_enabled.load(std::memory_order_relaxed) || !g_logger) {
    return;
  }

  // ---- Timestamp & metadata ----
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch())
                .count();
  uint64_t trace_id =
      g_trace_counter.fetch_add(1, std::memory_order_relaxed);
  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());

  const char *direction = is_host_to_device ? "H2D" : "D2H";
  const char *backend_str = backend_type_name(backend_type);

  // ---- Build JSON with snprintf (avoid fmt brace-escaping hell) ----
  std::string json;
  json.reserve(4096);
  char buf[256];

  // Metadata block
  snprintf(buf, sizeof(buf),
           "{\"trace_id\":%llu,\"ts_ns\":%lld,\"tid\":%lu,\"direction\":\"%s\","
           "\"backend\":\"%s\",\"num_blocks\":%d,\"start_layer_id\":%d,"
           "\"num_layers\":%d,\"is_mla\":%s,\"kv_dim\":%d,"
           "\"chunk_size_in_bytes\":%lld",
           (unsigned long long)trace_id, (long long)ns, (unsigned long)tid,
           direction, backend_str, num_blocks, start_layer_id, num_layers,
           is_mla ? "true" : "false", kv_dim,
           (long long)chunk_size_in_bytes);
  json += buf;

  // Strides
  snprintf(buf, sizeof(buf),
           ",\"strides\":{\"gpu_kv_stride\":%lld,\"gpu_block_stride\":%lld,"
           "\"gpu_layer_stride\":%lld,\"cpu_kv_stride\":%lld,"
           "\"cpu_layer_stride\":%lld,\"cpu_block_stride\":%lld}",
           (long long)gpu_kv_stride_in_bytes,
           (long long)gpu_block_stride_in_bytes,
           (long long)gpu_layer_stride_in_bytes,
           (long long)cpu_kv_stride_in_bytes,
           (long long)cpu_layer_stride_in_bytes,
           (long long)cpu_block_stride_in_bytes);
  json += buf;

  // Offsets
  snprintf(buf, sizeof(buf),
           ",\"offsets\":{\"gpu_startoff\":%lld,\"cpu_startoff\":%lld}",
           (long long)gpu_startoff_inside_chunks,
           (long long)cpu_startoff_inside_chunks);
  json += buf;

  // CE config
  snprintf(buf, sizeof(buf),
           ",\"ce_config\":{\"segment_threshold\":%d,\"path_opt_enabled\":%s,"
           "\"force_path\":%d,\"enable_memcpy2d\":%s,\"is_blockfirst\":%s,"
           "\"is_mla\":%s,\"batch_id\":%lld}",
           ce_config.segment_threshold,
           ce_config.path_opt_enabled ? "true" : "false",
           ce_config.force_path,
           ce_config.enable_memcpy2d ? "true" : "false",
           ce_config.is_blockfirst ? "true" : "false",
           ce_config.is_mla ? "true" : "false",
           (long long)ce_config.batch_id);
  json += buf;

  // CE analysis — booleans + segment count
  snprintf(buf, sizeof(buf),
           ",\"ce_analysis\":{\"gpu_log_contig\":%s,\"cpu_log_contig\":%s,"
           "\"cpu_phys_contig\":%s,\"gpu_phys_contig\":%s,\"num_segments\":%d,"
           "\"segments\":[",
           ce_analysis.gpu_log_contig ? "true" : "false",
           ce_analysis.cpu_log_contig ? "true" : "false",
           ce_analysis.cpu_phys_contig ? "true" : "false",
           ce_analysis.gpu_phys_contig ? "true" : "false",
           ce_analysis.num_segments);
  json += buf;

  // Segments array
  for (size_t i = 0; i < ce_analysis.segments.size(); ++i) {
    snprintf(buf, sizeof(buf),
             "{\"start_k\":%d,\"nr_blocks\":%d}%s",
             ce_analysis.segments[i].start_k,
             ce_analysis.segments[i].nr_blocks,
             (i + 1 < ce_analysis.segments.size()) ? "," : "");
    json += buf;
  }
  json += "]}";

  // CE path decision
  snprintf(buf, sizeof(buf),
           ",\"ce_path\":\"%s\",\"ce_path_id\":%d",
           ce_path_name(ce_path), static_cast<int>(ce_path));
  json += buf;

  // Block IDs (with truncation)
  int max_blocks = g_max_blocks;
  bool gpu_trunc = append_int64_array(json, ",\"gpu_block_ids\"",
                                      gpu_block_ids, num_blocks, max_blocks);
  bool cpu_trunc = append_int64_array(json, ",\"cpu_block_ids\"",
                                      cpu_block_ids, num_blocks, max_blocks);
  bool truncated = gpu_trunc || cpu_trunc;

  snprintf(buf, sizeof(buf),
           ",\"block_ids_truncated\":%s}",
           truncated ? "true" : "false");
  json += buf;

  // ---- Enqueue to spdlog async logger ----
  // The logger's background thread will handle the actual file write.
  // pattern is "%v" so only the raw message (our JSON) is written.
  g_logger->info(json);
}

}  // namespace flexkv
