"""CE transfer trace replay tool.

Reads a JSONL trace file produced by the C++ ``ce_trace_log()`` facility and
replays each H2D/D2H transfer through ``c_ext.transfer_kv_blocks()`` with
optional strategy overrides (``force_path``, ``path_opt``).

Typical workflow::

    # 1. Enable tracing during a production run
    export FLEXKV_CE_TRACE=1
    export FLEXKV_CE_TRACE_FILE=/tmp/ce_trace.jsonl
    # ... run FlexKV workload ...

    # 2. Summary of what was traced
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --summary

    # 3. Replay entry 0 with a different strategy
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --replay 0 --force-path 3

    # 4. Replay all entries and compare timings
    python -m scripts.ce_replay_cli /tmp/ce_trace.jsonl --replay-all --force-path 0
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from flexkv.common.debug import flexkv_logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CETraceEntry:
    """One trace record parsed from a JSONL line."""
    trace_id: int
    ts_ns: int
    tid: int
    direction: str
    backend: str
    num_blocks: int
    start_layer_id: int
    num_layers: int
    is_mla: bool
    kv_dim: int
    chunk_size_in_bytes: int
    # strides (all in bytes)
    gpu_kv_stride: int
    gpu_block_stride: int
    gpu_layer_stride: int
    cpu_kv_stride: int
    cpu_layer_stride: int
    cpu_block_stride: int
    # offsets
    gpu_startoff: int
    cpu_startoff: int
    # CE config
    ce_config: Dict[str, Any]
    # CE analysis
    ce_analysis: Dict[str, Any]
    # strategy
    ce_path: str
    ce_path_id: int
    # block IDs
    gpu_block_ids: List[int]
    cpu_block_ids: List[int]
    block_ids_truncated: bool

    @property
    def is_h2d(self) -> bool:
        return self.direction == "H2D"

    @property
    def backend_type_int(self) -> int:
        """Map backend name to the int expected by the binding."""
        return {"VLLM": 0, "TRTLLM": 1, "SGLANG": 2}.get(self.backend, 0)

    @property
    def batch_id(self) -> int:
        """Layerwise batch correlation ID (0 = standalone transfer)."""
        return self.ce_config.get("batch_id", 0)

    @property
    def transfer_bytes(self) -> int:
        """Total bytes moved in this transfer."""
        return self.num_blocks * self.num_layers * self.kv_dim * self.chunk_size_in_bytes

    @classmethod
    def from_json(cls, line: str) -> "CETraceEntry":
        d = json.loads(line)
        strides = d["strides"]
        offsets = d["offsets"]
        return cls(
            trace_id=d["trace_id"],
            ts_ns=d["ts_ns"],
            tid=d["tid"],
            direction=d["direction"],
            backend=d["backend"],
            num_blocks=d["num_blocks"],
            start_layer_id=d["start_layer_id"],
            num_layers=d["num_layers"],
            is_mla=d["is_mla"],
            kv_dim=d["kv_dim"],
            chunk_size_in_bytes=d["chunk_size_in_bytes"],
            gpu_kv_stride=strides["gpu_kv_stride"],
            gpu_block_stride=strides["gpu_block_stride"],
            gpu_layer_stride=strides["gpu_layer_stride"],
            cpu_kv_stride=strides["cpu_kv_stride"],
            cpu_layer_stride=strides["cpu_layer_stride"],
            cpu_block_stride=strides["cpu_block_stride"],
            gpu_startoff=offsets["gpu_startoff"],
            cpu_startoff=offsets["cpu_startoff"],
            ce_config=d["ce_config"],
            ce_analysis=d["ce_analysis"],
            ce_path=d["ce_path"],
            ce_path_id=d["ce_path_id"],
            gpu_block_ids=d["gpu_block_ids"],
            cpu_block_ids=d["cpu_block_ids"],
            block_ids_truncated=d.get("block_ids_truncated", False),
        )


@dataclass
class ReplayResult:
    """Result of a single replay."""
    trace_id: int
    direction: str
    original_path: str
    replayed_path: str
    elapsed_us: float
    bandwidth_gbs: float
    transfer_bytes: int
    num_blocks: int
    success: bool
    error: str = ""


# ---------------------------------------------------------------------------
# Tensor allocation helpers
# ---------------------------------------------------------------------------

def _allocate_gpu_tensors(
    entry: CETraceEntry,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Allocate GPU KV cache and return (ptr_tensor, keep_alive).

    The pointer layout depends on the backend:
      VLLM:   num_layers pointers (one per layer)
      TRTLLM: 1 pointer (shared, indexed by layer_stride)
      SGLANG: num_layers * kv_dim pointers (one per layer+kv pair)

    Callers must retain ``keep_alive`` until the transfer finishes; otherwise
    the underlying buffers may be freed while ``transfer_kv_blocks`` still
    holds raw pointers.
    """
    max_gpu_id = max(entry.gpu_block_ids) if entry.gpu_block_ids else 0
    # Each GPU buffer must hold (max_gpu_id + 1) blocks.
    gpu_buf_bytes = (max_gpu_id + 1) * entry.gpu_block_stride
    # Round up to int64 elements for tensor allocation.
    gpu_buf_elems = (gpu_buf_bytes + 7) // 8

    if entry.backend == "VLLM":
        num_ptrs = entry.num_layers
    elif entry.backend == "TRTLLM":
        num_ptrs = 1
    elif entry.backend == "SGLANG":
        num_ptrs = entry.num_layers * entry.kv_dim
    else:
        num_ptrs = entry.num_layers

    keep_alive: List[torch.Tensor] = []
    for _ in range(num_ptrs):
        gpu_tensor = torch.empty(gpu_buf_elems, dtype=torch.int64, device="cuda")
        keep_alive.append(gpu_tensor)

    # Store pointers as CPU int64 tensor.
    ptr_tensor = torch.tensor(
        [t.data_ptr() for t in keep_alive], dtype=torch.int64
    )
    return ptr_tensor, keep_alive


def _allocate_cpu_tensor(entry: CETraceEntry) -> torch.Tensor:
    """Allocate a CPU KV cache tensor (pinned for async DMA)."""
    max_cpu_id = max(entry.cpu_block_ids) if entry.cpu_block_ids else 0
    # CPU addressing uses both block and layer strides; size for the max of
    # the two common layouts (block-major vs layer-strided).
    by_blocks = (max_cpu_id + 2) * entry.cpu_block_stride
    by_layers = (
        (entry.start_layer_id + entry.num_layers + 1)
        * entry.cpu_layer_stride
        * max(entry.kv_dim, 1)
        + (max_cpu_id + 2) * entry.chunk_size_in_bytes
    )
    cpu_buf_bytes = max(by_blocks, by_layers)
    cpu_buf_elems = (cpu_buf_bytes + 7) // 8
    cpu_tensor = torch.empty(cpu_buf_elems, dtype=torch.int64, pin_memory=True)
    return cpu_tensor


# ---------------------------------------------------------------------------
# Main replayer
# ---------------------------------------------------------------------------

def compact_block_ids(entry: CETraceEntry) -> CETraceEntry:
    """Remap GPU/CPU block IDs into a dense ``0..N-1`` range.

    Production traces often use sparse high block IDs (CPU ids in the tens of
    thousands).  Naive allocation of ``(max_id+1) * cpu_block_stride`` can
    request hundreds of GiB of pinned memory and appear to hang.  Compacting
    preserves transfer *shape* (N blocks, strides, path) for stress replay
    while keeping buffers tractable.

    Also forces ``start_layer_id=0`` so VLLM ``ptr_at(layer_idx)`` indexes the
    compact pointer array of length ``num_layers`` (layerwise traces often have
    ``start_layer_id`` in 0..L-1 with ``num_layers==1``).
    """
    n = len(entry.gpu_block_ids)
    if n == 0:
        return entry
    entry.gpu_block_ids = list(range(n))
    entry.cpu_block_ids = list(range(n))
    entry.num_blocks = n
    entry.block_ids_truncated = False
    entry.start_layer_id = 0
    return entry


class CETraceReplayer:
    """Read a CE trace file and replay transfers with optional overrides."""

    def __init__(self, trace_file: str, compact_ids: bool = False):
        self.trace_file = trace_file
        self.compact_ids = compact_ids
        self.entries: List[CETraceEntry] = []
        self._load()

    def _load(self) -> None:
        with open(self.trace_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = CETraceEntry.from_json(line)
                    if self.compact_ids:
                        entry = compact_block_ids(entry)
                    self.entries.append(entry)
                except (json.JSONDecodeError, KeyError) as e:
                    flexkv_logger.warning(
                        f"[ce_replay] skipping malformed trace line: {e}"
                    )

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the trace contents."""
        if not self.entries:
            return {"total_entries": 0}

        path_counts: Dict[str, int] = {}
        dir_counts: Dict[str, int] = {}
        batch_ids = set()
        total_bytes = 0
        for e in self.entries:
            path_counts[e.ce_path] = path_counts.get(e.ce_path, 0) + 1
            dir_counts[e.direction] = dir_counts.get(e.direction, 0) + 1
            batch_ids.add(e.batch_id)
            total_bytes += e.transfer_bytes

        return {
            "total_entries": len(self.entries),
            "directions": dir_counts,
            "strategy_distribution": path_counts,
            "total_transfer_bytes": total_bytes,
            "total_transfer_gb": total_bytes / (1024**3),
            "truncated_entries": sum(
                1 for e in self.entries if e.block_ids_truncated
            ),
            "num_batches": len(batch_ids) - (1 if 0 in batch_ids else 0),
            "time_span_ns": (
                self.entries[-1].ts_ns - self.entries[0].ts_ns
                if len(self.entries) > 1
                else 0
            ),
        }

    def replay(
        self,
        entry_idx: int = -1,
        force_path: int = -1,
        path_opt: Optional[bool] = None,
        segment_threshold: Optional[int] = None,
        sync: bool = True,
    ) -> ReplayResult:
        """Replay a single trace entry.

        Args:
            entry_idx: Index into self.entries (-1 = last).
            force_path: Override strategy (-1 = use original, 0-4 = force).
            path_opt: Override path_opt_enabled (None = use original).
            segment_threshold: Override segment threshold (None = use original).
            sync: Whether to synchronize after transfer.

        Returns:
            ReplayResult with timing and bandwidth.
        """
        from flexkv import c_ext

        entry = self.entries[entry_idx]

        # Determine override params
        ce_path_opt = (
            path_opt if path_opt is not None
            else entry.ce_config["path_opt_enabled"]
        )
        ce_force_path = force_path if force_path >= 0 else entry.ce_config["force_path"]
        ce_threshold = (
            segment_threshold if segment_threshold is not None
            else entry.ce_config["segment_threshold"]
        )

        # Determine the replayed path name
        if not ce_path_opt:
            replayed_path = "PER_BLOCK"
        elif ce_force_path >= 0:
            path_names = [
                "CONTIG_DIRECT", "SEGMENT_DIRECT", "SEGMENT_SCATTER",
                "GATHER_SCATTER", "GATHER_DIRECT",
            ]
            replayed_path = path_names[ce_force_path] if ce_force_path <= 4 else "UNKNOWN"
        else:
            replayed_path = entry.ce_path

        # Warn about truncated traces
        if entry.block_ids_truncated:
            flexkv_logger.warning(
                f"[ce_replay] trace_id={entry.trace_id} has truncated block "
                f"IDs — replaying with {len(entry.gpu_block_ids)} blocks "
                f"(original had {entry.num_blocks})"
            )

        # Allocate tensors
        gpu_block_id_tensor = torch.tensor(
            entry.gpu_block_ids, dtype=torch.int64
        ).pin_memory()
        cpu_block_id_tensor = torch.tensor(
            entry.cpu_block_ids, dtype=torch.int64
        ).pin_memory()
        gpu_tensor_ptrs_tensor, gpu_keep_alive = _allocate_gpu_tensors(entry)
        cpu_tensor = _allocate_cpu_tensor(entry)

        # Adjust num_blocks if truncated
        num_blocks = len(entry.gpu_block_ids)

        # Time the transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter_ns()

        try:
            c_ext.transfer_kv_blocks(
                gpu_block_id_tensor=gpu_block_id_tensor,
                gpu_tensor_ptrs_tensor=gpu_tensor_ptrs_tensor,
                gpu_kv_stride_in_bytes=entry.gpu_kv_stride,
                gpu_block_stride_in_bytes=entry.gpu_block_stride,
                gpu_layer_stride_in_bytes=entry.gpu_layer_stride,
                cpu_block_id_tensor=cpu_block_id_tensor,
                cpu_tensor=cpu_tensor,
                cpu_kv_stride_in_bytes=entry.cpu_kv_stride,
                cpu_layer_stride_in_bytes=entry.cpu_layer_stride,
                cpu_block_stride_in_bytes=entry.cpu_block_stride,
                chunk_size_in_bytes=entry.chunk_size_in_bytes,
                start_layer_id=entry.start_layer_id,
                num_layers=entry.num_layers,
                transfer_num_cta=4,
                is_host_to_device=entry.is_h2d,
                use_ce_transfer=True,
                is_mla=entry.is_mla,
                gpu_block_type=entry.backend_type_int,
                sync=sync,
                ce_path_opt=ce_path_opt,
                ce_segment_threshold=ce_threshold,
                ce_force_path=ce_force_path,
                ce_enable_memcpy2d=entry.ce_config["enable_memcpy2d"],
                is_blockfirst=entry.ce_config["is_blockfirst"],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_us = elapsed_ns / 1000.0
            transfer_bytes = num_blocks * entry.num_layers * entry.kv_dim * entry.chunk_size_in_bytes
            bandwidth_gbs = (transfer_bytes / elapsed_ns) * 1e9 if elapsed_ns > 0 else 0.0
            # Keep buffers alive until after sync (prevent use-after-free).
            _ = gpu_keep_alive
            return ReplayResult(
                trace_id=entry.trace_id,
                direction=entry.direction,
                original_path=entry.ce_path,
                replayed_path=replayed_path,
                elapsed_us=elapsed_us,
                bandwidth_gbs=bandwidth_gbs,
                transfer_bytes=transfer_bytes,
                num_blocks=num_blocks,
                success=True,
            )
        except Exception as e:
            elapsed_ns = time.perf_counter_ns() - start
            _ = gpu_keep_alive
            return ReplayResult(
                trace_id=entry.trace_id,
                direction=entry.direction,
                original_path=entry.ce_path,
                replayed_path=replayed_path,
                elapsed_us=(elapsed_ns / 1000.0),
                bandwidth_gbs=0.0,
                transfer_bytes=0,
                num_blocks=num_blocks,
                success=False,
                error=str(e),
            )

    def replay_all(
        self,
        force_path: int = -1,
        path_opt: Optional[bool] = None,
        segment_threshold: Optional[int] = None,
    ) -> List[ReplayResult]:
        """Replay all entries and return results."""
        results = []
        for i in range(len(self.entries)):
            results.append(self.replay(
                entry_idx=i,
                force_path=force_path,
                path_opt=path_opt,
                segment_threshold=segment_threshold,
            ))
        return results

    def compare_strategies(
        self,
        entry_idx: int = -1,
    ) -> List[ReplayResult]:
        """Replay one entry with all strategies and return comparison."""
        results = []
        # Original (auto-select)
        results.append(self.replay(entry_idx=entry_idx))
        # Each forced path (0-4: CONTIG_DIRECT..GATHER_DIRECT)
        for path_id in range(5):
            r = self.replay(entry_idx=entry_idx, force_path=path_id)
            results.append(r)
        # PER_BLOCK baseline
        r = self.replay(entry_idx=entry_idx, path_opt=False)
        results.append(r)
        return results
