# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlexKV RadixCache replacement for SGLang.

Unlike the HiCacheStorage adapter (L3 backend under HiRadixCache), this
class directly replaces RadixCache and uses FlexKV's native GPU transfer
engine (get_async/put_async with slot_mapping) for GPU<->CPU data movement,
eliminating the double-memcpy through SGLang's host memory pool.

Architecture
------------
    SGLang Scheduler
         |
    FlexKVRadixCache (extends RadixCache)
     /                 \\
  SGLang RadixTree    FlexKV KVManager
  (GPU slot tracking) (CPU/SSD/Remote cache + transfer)
     \\                 /
  GPU KV Buffers (k_buffer[layer], v_buffer[layer])
  (registered with FlexKV TransferEngine)

Data flow (GET / match_prefix):
  1. RadixCache.match_prefix -> GPU-cached prefix
  2. FlexKV get_match -> check FlexKV cache for uncached tail
  3. Allocate GPU slots, launch transfer, wait
  4. Create new RadixTree nodes for loaded data

Data flow (PUT / cache_finished_req):
  1. super().cache_finished_req -> insert into SGLang RadixTree
  2. FlexKV put_match -> dedup check
  3. Async launch GPU->CPU transfer for new tokens
  4. Track in-flight transfers for eviction safety
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig as SGLangModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[FlexKVRadixCache] %(levelname)s %(message)s"))
    logger.addHandler(_handler)
    _log_level = os.environ.get("FLEXKV_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, _log_level, logging.INFO))


def _check_success(responses, task_id) -> bool:
    """Check if a FlexKV transfer completed successfully."""
    from flexkv.common.request import KVResponseStatus

    if not responses or task_id not in responses:
        return False
    return responses[task_id].status == KVResponseStatus.SUCCESS


class FlexKVRadixCache(RadixCache):
    """RadixCache + FlexKV GPU<->CPU transfer.

    This subclass adds:
      - FlexKV KVManager with process-mode TransferEngine for GPU<->CPU transfers
      - GPU KV cache registration via KVTPClient (same pattern as vLLM adapter)
      - Overridden ``match_prefix`` to fetch missing prefix from FlexKV
      - Overridden ``cache_finished_req`` to asynchronously store KV to FlexKV
      - Overridden ``evict`` to wait for in-flight transfers before eviction
      - Local and distributed mode support (Redis GMS + P2P for cross-node)
    """

    def __init__(
        self,
        params: "CacheInitParams",
        model_config: Optional["SGLangModelConfig"] = None,
        tp_size: int = 1,
        rank: int = 0,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(params)

        self._tp_size = tp_size
        self._rank = rank
        self._tp_group = tp_group

        # In-flight PUT tracking for eviction safety
        self._in_flight_put_tasks: Dict[int, TreeNode] = {}
        self._node_lock = threading.Lock()

        # Timeouts
        self._get_timeout = float(os.environ.get("FLEXKV_GET_TIMEOUT", "20.0"))
        self._put_timeout = float(os.environ.get("FLEXKV_PUT_TIMEOUT", "20.0"))
        self._prefetch_timeout = float(os.environ.get("FLEXKV_PREFETCH_TIMEOUT", "5.0"))

        # Initialize FlexKV
        self._kv_manager = None
        self._tp_client = None
        self._mode = "local"  # or "distributed"
        self._init_flexkv(model_config, tp_size, rank)

        logger.info(
            "FlexKVRadixCache initialized (mode=%s, page_size=%d, rank=%d, tp=%d)",
            self._mode, self.page_size, rank, tp_size,
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_flexkv(
        self,
        sglang_model_config: Optional["SGLangModelConfig"],
        tp_size: int,
        rank: int,
    ):
        """Create FlexKV KVManager in process mode and register GPU KV cache."""
        from flexkv.integration.config import FlexKVConfig
        from flexkv.kvmanager import KVManager

        # Load config from env / config file
        flexkv_config = FlexKVConfig.from_env()

        # Populate model params from SGLang config
        if sglang_model_config is not None:
            flexkv_config.post_init_from_sglang_config(
                sglang_model_config, tp_size, self.page_size,
            )
        else:
            # Fallback: set tokens_per_block from page_size
            flexkv_config.cache_config.tokens_per_block = self.page_size

        # Parse mode from env or config
        self._mode = os.environ.get(
            "FLEXKV_MODE",
            flexkv_config.user_config.__dict__.get("mode", "local")
            if hasattr(flexkv_config.user_config, "__dict__") else "local"
        )
        if self._mode not in ("local", "distributed"):
            raise ValueError(f"Invalid FLEXKV_MODE: {self._mode}")

        # Configure distributed mode
        if self._mode == "distributed":
            redis_host = os.environ.get("FLEXKV_REDIS_HOST", "127.0.0.1")
            redis_port = int(os.environ.get("FLEXKV_REDIS_PORT", "6379"))
            redis_password = os.environ.get("FLEXKV_REDIS_PASSWORD", None)
            flexkv_config.cache_config.enable_remote = True
            flexkv_config.cache_config.enable_kv_sharing = True
            flexkv_config.cache_config.enable_p2p_cpu = True
            flexkv_config.cache_config.enable_p2p_ssd = flexkv_config.cache_config.enable_ssd
            flexkv_config.cache_config.redis_host = redis_host
            flexkv_config.cache_config.redis_port = redis_port
            flexkv_config.cache_config.redis_password = redis_password
            logger.info(
                "Distributed mode: Redis=%s:%d, P2P CPU, P2P SSD=%s",
                redis_host, redis_port, flexkv_config.cache_config.enable_ssd,
            )
        else:
            flexkv_config.cache_config.enable_remote = False
            flexkv_config.cache_config.enable_kv_sharing = False
            flexkv_config.cache_config.enable_p2p_cpu = False
            flexkv_config.cache_config.enable_p2p_ssd = False

        # Ensure process mode (NOT CPU-only) for GPU transfer support
        if os.environ.get("FLEXKV_CPU_ONLY") == "1":
            logger.warning(
                "FLEXKV_CPU_ONLY=1 is incompatible with FlexKVRadixCache "
                "(need GPU transfers). Unsetting it."
            )
            os.environ.pop("FLEXKV_CPU_ONLY", None)

        # Create KVManager
        self._kv_manager = KVManager(
            model_config=flexkv_config.model_config,
            cache_config=flexkv_config.cache_config,
        )
        self._kv_manager.start()

        # Store gpu_register_port for KVTPClient registration
        self._gpu_register_port = flexkv_config.gpu_register_port

        # Register GPU KV cache buffers with the TransferManager subprocess
        self._register_gpu_kv_cache()

        # Wait for KVManager to become ready (GPU registration complete)
        import time
        deadline = time.monotonic() + 30.0
        while not self._kv_manager.is_ready():
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "FlexKV KVManager did not become ready within 30s. "
                    "GPU registration may have failed."
                )
            time.sleep(0.1)

        logger.info("FlexKV KVManager ready (mode=%s)", self._mode)

    def _register_gpu_kv_cache(self):
        """Register SGLang's GPU KV buffers with FlexKV TransferEngine.

        Uses KVTPClient to send CUDA IPC tensor handles to the TransferManager
        subprocess, following the same pattern as the vLLM adapter.
        """
        from flexkv.server.client import KVTPClient
        from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        device_id = torch.cuda.current_device()

        # Detect MHA vs MLA
        is_mla = hasattr(kvcache, "kv_buffer") and not hasattr(kvcache, "k_buffer")

        if is_mla:
            # MLA: single kv_buffer per layer
            gpu_tensors = list(kvcache.kv_buffer)
            num_layers = len(gpu_tensors)
            num_slots = gpu_tensors[0].shape[0]
            num_kv_heads = 1
            head_size = gpu_tensors[0].shape[-1]
        else:
            # MHA: interleave k_buffer and v_buffer per layer
            # SGLang shape per layer: [total_slots, head_num, head_dim]
            gpu_tensors = []
            for layer_idx in range(len(kvcache.k_buffer)):
                gpu_tensors.append(kvcache.k_buffer[layer_idx])
                gpu_tensors.append(kvcache.v_buffer[layer_idx])
            num_layers = len(kvcache.k_buffer)
            num_slots = kvcache.k_buffer[0].shape[0]
            num_kv_heads = kvcache.head_num
            head_size = kvcache.head_dim

        # SGLang's k_buffer[layer] is [total_slots, heads, dim] — a flat
        # per-token buffer with NO block structure.  Register with
        # tokens_per_block=1 so each GPU "block" is exactly one token slot.
        # This way slot_mapping values (token-level indices) map 1:1 to
        # block IDs, and the TransferEngine computes correct offsets.
        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layers,
            num_block=num_slots,
            tokens_per_block=1,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=is_mla,
        )

        tp_client = KVTPClient(
            gpu_register_port=self._gpu_register_port,
            dp_client_id=0,
            device_id=device_id,
        )
        tp_client.register_to_server(gpu_tensors, gpu_layout)
        self._tp_client = tp_client

        logger.info(
            "GPU KV cache registered: layers=%d, slots=%d (tokens_per_block=1), "
            "heads=%d, head_dim=%d, mla=%s",
            num_layers, num_slots,
            num_kv_heads, head_size, is_mla,
        )

    # ------------------------------------------------------------------
    # GET path: match_prefix override
    # ------------------------------------------------------------------

    def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
        """Match prefix in GPU RadixTree, then try FlexKV for uncached tail."""
        if self.disable or not key:
            return super().match_prefix(key, **kwargs)

        # Page-align the key
        if self.page_size != 1:
            aligned_len = len(key) // self.page_size * self.page_size
            key = key[:aligned_len]

        if len(key) == 0:
            return super().match_prefix(key, **kwargs)

        # Step 1: GPU-level match
        base_res = super().match_prefix(key, **kwargs)
        value = base_res.device_indices
        last_node = base_res.last_device_node

        # Already fully matched in GPU
        if value.numel() >= len(key):
            return base_res

        # Step 2: Try FlexKV for uncached tail
        uncached_start = value.numel()
        try:
            result = self._flexkv_get_uncached(key, uncached_start, last_node)
        except Exception:
            logger.exception("FlexKV GET failed, returning base match")
            return base_res

        if result is None:
            return base_res

        extended_value, new_last_node = result
        return MatchResult(
            device_indices=torch.cat([value, extended_value]),
            last_device_node=new_last_node,
            last_host_node=new_last_node,
        )

    def _flexkv_get_uncached(
        self,
        full_key: RadixKey,
        base_matched_len: int,
        last_node: TreeNode,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        """Try to load the uncached suffix from FlexKV into GPU.

        Returns (new_device_indices, new_last_node) or None on miss/failure.
        """
        token_ids = full_key.token_ids
        np_token_ids = np.array(token_ids, dtype=np.int64)

        # In distributed mode, prefetch remote blocks first
        if self._mode == "distributed":
            self._prefetch_remote(np_token_ids)

        # Build mask: only care about uncached tail
        token_mask = np.zeros(len(token_ids), dtype=np.bool_)
        token_mask[base_matched_len:] = True

        # Phase 1: Match against FlexKV radix tree
        task_id, return_mask = self._kv_manager.get_match(
            token_ids=np_token_ids,
            token_mask=token_mask,
        )

        # return_mask[i] == True means FlexKV has token i cached
        # Count consecutive matched tokens from base_matched_len
        num_flexkv_matched = 0
        for i in range(base_matched_len, len(return_mask)):
            if return_mask[i]:
                num_flexkv_matched += 1
            else:
                break

        if num_flexkv_matched == 0:
            # get_match task is already completed internally, no need to cancel
            return None

        # Page-align the matched count
        num_to_load = (num_flexkv_matched // self.page_size) * self.page_size
        if num_to_load == 0:
            return None

        # Phase 2: Allocate GPU slots
        if self.token_to_kv_pool_allocator.available_size() < num_to_load:
            self.evict(num_to_load)

        token_slots = self.token_to_kv_pool_allocator.alloc(num_to_load)
        if token_slots is None:
            logger.debug("GPU slot allocation failed for %d tokens", num_to_load)
            return None

        # Phase 3: Build slot_mapping and launch transfer
        slot_mapping = token_slots.cpu().numpy().astype(np.int64)
        self._kv_manager.launch([task_id], [slot_mapping])

        # Phase 4: Wait for transfer completion
        responses = self._kv_manager.wait([task_id], timeout=self._get_timeout)
        if not _check_success(responses, task_id):
            self.token_to_kv_pool_allocator.free(token_slots)
            logger.debug("FlexKV GET transfer failed for task %d", task_id)
            return None

        # Phase 5: Create RadixTree node for loaded data
        new_node = TreeNode(priority=last_node.priority)
        start = base_matched_len
        end = start + num_to_load
        new_node.key = full_key[start:end]
        new_node.value = token_slots[:num_to_load]
        new_node.parent = last_node
        last_node.children[self.get_child_key_fn(new_node.key)] = new_node
        self.evictable_size_ += num_to_load

        logger.debug(
            "FlexKV GET: loaded %d tokens (pos %d-%d)",
            num_to_load, start, end,
        )
        return (token_slots[:num_to_load], new_node)

    def _prefetch_remote(self, token_ids: np.ndarray):
        """In distributed mode, prefetch remote blocks into local CPU cache."""
        from flexkv.common.request import KVResponseStatus

        try:
            task_id = self._kv_manager.prefetch_async(token_ids=token_ids)
            responses = self._kv_manager.wait(
                [task_id], timeout=self._prefetch_timeout
            )
            if not responses or task_id not in responses:
                logger.debug("Remote prefetch timeout")
                return
            resp = responses[task_id]
            if resp.status != KVResponseStatus.SUCCESS:
                logger.debug("Remote prefetch status=%s", resp.status.value)
        except Exception:
            logger.debug("Remote prefetch failed", exc_info=True)

    # ------------------------------------------------------------------
    # PUT path: cache_finished_req override
    # ------------------------------------------------------------------

    def cache_finished_req(self, req: "Req", is_insert: bool = True) -> None:
        """Insert into RadixTree, then asynchronously store KV to FlexKV."""
        if not is_insert or self._kv_manager is None:
            super().cache_finished_req(req, is_insert=is_insert)
            return

        # Capture data BEFORE super() frees req_pool_idx and kv_indices.
        # RadixCache uses page_align(fill_ids) as keys.
        fill_ids = list(req.fill_ids)
        aligned_len = (len(fill_ids) // self.page_size) * self.page_size
        if aligned_len > 0:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :aligned_len
            ].to(dtype=torch.int64, copy=True)
        else:
            kv_indices = None

        super().cache_finished_req(req, is_insert=is_insert)

        if kv_indices is None:
            return

        try:
            self._flexkv_put_async(
                fill_ids[:aligned_len], kv_indices, req.last_node,
            )
        except Exception:
            logger.exception("FlexKV PUT failed")

    def _flexkv_put_async(
        self,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        last_node: Optional[TreeNode],
    ):
        """Asynchronously store KV from GPU to FlexKV cache.

        Args:
            token_ids: Page-aligned token IDs to store.
            kv_indices: GPU KV cache slot indices (captured before super() freed them).
            last_node: The RadixTree node to lock during transfer.
        """
        np_token_ids = np.array(token_ids, dtype=np.int64)

        # Check what FlexKV already has (dedup)
        task_id, return_mask = self._kv_manager.put_match(
            token_ids=np_token_ids,
        )

        # return_mask[i] == True means token i needs GPU->CPU transfer
        num_to_transfer = int(return_mask.sum())
        if num_to_transfer == 0:
            return

        # Build slot_mapping for tokens that need transfer
        transfer_positions = np.where(return_mask)[0]
        slot_mapping = kv_indices[transfer_positions].cpu().numpy().astype(np.int64)

        # Launch async GPU->CPU transfer
        self._kv_manager.launch([task_id], [slot_mapping])

        # Track in-flight for eviction safety
        if last_node is not None and last_node is not self.root_node:
            self.inc_lock_ref(last_node)
            with self._node_lock:
                self._in_flight_put_tasks[task_id] = last_node

        logger.debug(
            "FlexKV PUT: storing %d/%d tokens async (task=%d)",
            num_to_transfer, len(token_ids), task_id,
        )

    # ------------------------------------------------------------------
    # Eviction: non-blocking poll + fallback blocking wait
    # ------------------------------------------------------------------

    def evict(self, num_tokens: int) -> None:
        """Poll completed PUT transfers, evict, block-wait only if needed."""
        if self.disable:
            return

        # Step 1: Non-blocking poll to release completed PUT locks
        self._poll_completed_puts()

        # Step 2: Try eviction with currently evictable nodes
        super().evict(num_tokens)

        # Step 3: If still not enough, block-wait remaining PUTs and retry
        if (self.token_to_kv_pool_allocator.available_size() < num_tokens
                and self._in_flight_put_tasks):
            self._block_wait_all_puts()
            super().evict(num_tokens)

    def _poll_completed_puts(self):
        """Non-blocking poll: release locks for completed PUT transfers."""
        with self._node_lock:
            if not self._in_flight_put_tasks:
                return
            task_ids = list(self._in_flight_put_tasks.keys())

        if not task_ids:
            return

        try:
            completed = self._kv_manager.try_wait(task_ids)
        except Exception:
            logger.exception("Failed to poll in-flight PUT tasks")
            return

        if not completed:
            return

        with self._node_lock:
            for task_id in completed:
                node = self._in_flight_put_tasks.pop(task_id, None)
                if node is not None:
                    self.dec_lock_ref(node)

    def _block_wait_all_puts(self):
        """Blocking wait for ALL in-flight PUT transfers and release locks."""
        with self._node_lock:
            if not self._in_flight_put_tasks:
                return
            task_ids = list(self._in_flight_put_tasks.keys())

        if not task_ids:
            return

        try:
            self._kv_manager.wait(task_ids, timeout=self._put_timeout)
        except Exception:
            logger.exception("Failed to wait for in-flight PUT tasks")

        with self._node_lock:
            for task_id in task_ids:
                node = self._in_flight_put_tasks.pop(task_id, None)
                if node is not None:
                    self.dec_lock_ref(node)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def reset(self):
        super().reset()
        if hasattr(self, "_in_flight_put_tasks"):
            with self._node_lock:
                self._in_flight_put_tasks.clear()

    def shutdown(self):
        """Shutdown FlexKV KVManager and release resources."""
        self._block_wait_all_puts()
        if self._kv_manager is not None:
            try:
                self._kv_manager.shutdown()
            except Exception:
                logger.exception("KVManager shutdown failed")
            self._kv_manager = None
        logger.info("FlexKVRadixCache shut down")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
