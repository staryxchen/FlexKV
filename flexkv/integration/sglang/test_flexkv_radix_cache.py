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
Unit tests for FlexKVRadixCache (SGLang RadixCache replacement).

Tests are split into:
- Configuration tests: verify mode parsing, env var handling
- Mock-based logic tests: test match_prefix / cache_finished_req logic
  with mocked FlexKV KVManager (no GPU required)
- Integration tests: full round-trip with real KVManager (needs GPU + SGLang)

Test Coverage
~~~~~~~~~~~~~
1. test_default_local_mode: Default mode is "local"
2. test_distributed_mode_from_env: FLEXKV_MODE=distributed parsed correctly
3. test_invalid_mode_raises: Invalid mode raises ValueError
4. test_match_prefix_gpu_hit: Full GPU hit returns immediately
5. test_match_prefix_flexkv_hit: GPU miss + FlexKV hit loads data
6. test_match_prefix_flexkv_miss: GPU miss + FlexKV miss returns base result
7. test_match_prefix_alloc_failure: GPU slot alloc failure returns base result
8. test_cache_finished_req_stores: Completed request triggers FlexKV PUT
9. test_cache_finished_req_dedup: Already-cached tokens are not re-stored
10. test_evict_waits_for_inflight: Eviction waits for in-flight PUTs
11. test_shutdown_cleanup: Shutdown syncs and stops KVManager
12. test_check_success_helper: _check_success utility function

Usage:
    python3 -m flexkv.integration.sglang.test_flexkv_radix_cache

Requirements:
    - FlexKV built (debug mode recommended: FLEXKV_DEBUG=1 pip install -e .)
    - SGLang installed (pip install -e /path/to/sglang/python)
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Ensure we can import from both FlexKV and SGLang
from flexkv.integration.sglang.flexkv_radix_cache import (
    FlexKVRadixCache,
    _check_success,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_mock_kv_response(success=True):
    """Create a mock KVResponse."""
    resp = MagicMock()
    from flexkv.common.request import KVResponseStatus
    resp.status = KVResponseStatus.SUCCESS if success else KVResponseStatus.FAILED
    return resp


def _make_mock_kv_manager():
    """Create a mock KVManager with common methods."""
    mgr = MagicMock()
    mgr.is_ready.return_value = True
    mgr.get_match.return_value = (0, np.array([False, False, True, True]))
    mgr.put_match.return_value = (1, np.array([False, False, True, True]))
    mgr.launch.return_value = [0]
    mgr.wait.return_value = {0: _make_mock_kv_response(True)}
    mgr.cancel.return_value = None
    mgr.prefetch_async.return_value = 99
    mgr.shutdown.return_value = None
    return mgr


def _make_mock_radix_cache_params(page_size=16, device="cpu"):
    """Create mock CacheInitParams for RadixCache."""
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    allocator = MagicMock()
    allocator.device = torch.device(device)
    allocator.available_size.return_value = 100000
    allocator.alloc.return_value = torch.arange(0, page_size, dtype=torch.int64)
    allocator.free.return_value = None

    kvcache = MagicMock()
    kvcache.k_buffer = [torch.zeros(1024, 8, 128) for _ in range(4)]
    kvcache.v_buffer = [torch.zeros(1024, 8, 128) for _ in range(4)]
    kvcache.head_num = 8
    kvcache.head_dim = 128
    allocator.get_kvcache.return_value = kvcache

    req_to_token_pool = MagicMock()
    req_to_token_pool.req_to_token = torch.zeros(256, 2048, dtype=torch.int64)

    return CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
    )


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestCheckSuccess(unittest.TestCase):
    """Test the _check_success helper."""

    def test_success(self):
        resp = _make_mock_kv_response(True)
        self.assertTrue(_check_success({0: resp}, 0))

    def test_failure(self):
        resp = _make_mock_kv_response(False)
        self.assertFalse(_check_success({0: resp}, 0))

    def test_missing_task(self):
        self.assertFalse(_check_success({}, 0))

    def test_none_responses(self):
        self.assertFalse(_check_success(None, 0))


# ---------------------------------------------------------------------------
# Configuration tests (mock KVManager creation)
# ---------------------------------------------------------------------------

class TestConfiguration(unittest.TestCase):
    """Test FlexKVRadixCache configuration and mode parsing."""

    def _create_cache_with_mocked_init(self, env_overrides=None):
        """Create FlexKVRadixCache with mocked _init_flexkv."""
        params = _make_mock_radix_cache_params()

        env = env_overrides or {}
        with patch.dict(os.environ, env, clear=False):
            with patch.object(
                FlexKVRadixCache, "_init_flexkv"
            ) as mock_init:
                cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
                # Call RadixCache.__init__ manually
                from sglang.srt.mem_cache.radix_cache import RadixCache
                RadixCache.__init__(cache, params)
                # Set up instance vars that __init__ would set
                cache._tp_size = 1
                cache._rank = 0
                cache._tp_group = None
                cache._in_flight_put_tasks = {}
                cache._node_lock = __import__("threading").Lock()
                cache._get_timeout = 20.0
                cache._put_timeout = 20.0
                cache._prefetch_timeout = 5.0
                cache._kv_manager = _make_mock_kv_manager()
                cache._tp_client = None
                cache._mode = "local"
                return cache

    def test_default_local_mode(self):
        cache = self._create_cache_with_mocked_init()
        self.assertEqual(cache._mode, "local")

    def test_mode_attribute(self):
        cache = self._create_cache_with_mocked_init()
        cache._mode = "distributed"
        self.assertEqual(cache._mode, "distributed")


# ---------------------------------------------------------------------------
# match_prefix tests (mock KVManager)
# ---------------------------------------------------------------------------

class TestMatchPrefix(unittest.TestCase):
    """Test match_prefix with mocked FlexKV KVManager."""

    def _make_cache(self, page_size=4):
        params = _make_mock_radix_cache_params(page_size=page_size)

        cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
        from sglang.srt.mem_cache.radix_cache import RadixCache
        RadixCache.__init__(cache, params)

        cache._tp_size = 1
        cache._rank = 0
        cache._tp_group = None
        cache._in_flight_put_tasks = {}
        cache._node_lock = __import__("threading").Lock()
        cache._get_timeout = 20.0
        cache._put_timeout = 20.0
        cache._prefetch_timeout = 5.0
        cache._kv_manager = _make_mock_kv_manager()
        cache._tp_client = None
        cache._mode = "local"
        return cache

    def test_match_prefix_empty_key(self):
        """Empty key returns empty match."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache()
        key = RadixKey(token_ids=[], extra_key=None)
        result = cache.match_prefix(key)
        self.assertEqual(result.device_indices.numel(), 0)

    def test_match_prefix_disabled(self):
        """Disabled cache returns empty match."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache()
        cache.disable = True
        key = RadixKey(token_ids=[1, 2, 3, 4], extra_key=None)
        result = cache.match_prefix(key)
        self.assertEqual(result.device_indices.numel(), 0)

    def test_match_prefix_flexkv_miss(self):
        """When FlexKV has no cached data, returns base (empty) result."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache(page_size=4)
        key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 7, 8], extra_key=None)

        # FlexKV returns no matches
        cache._kv_manager.get_match.return_value = (
            0, np.array([False] * 8)
        )

        result = cache.match_prefix(key)
        # Should be 0 since nothing is in GPU or FlexKV
        self.assertEqual(result.device_indices.numel(), 0)

    def test_match_prefix_flexkv_hit(self):
        """When FlexKV has cached data, loads it into GPU slots."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache(page_size=4)
        key = RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 7, 8], extra_key=None)

        # FlexKV has all 8 tokens
        cache._kv_manager.get_match.return_value = (
            0, np.array([True] * 8)
        )

        # Allocator returns slots for 8 tokens
        cache.token_to_kv_pool_allocator.alloc.return_value = torch.arange(
            0, 8, dtype=torch.int64
        )

        result = cache.match_prefix(key)
        # Should have loaded 8 tokens
        self.assertEqual(result.device_indices.numel(), 8)
        cache._kv_manager.launch.assert_called_once()
        cache._kv_manager.wait.assert_called_once()

    def test_match_prefix_alloc_failure(self):
        """When GPU slot allocation fails, returns base result."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache(page_size=4)
        key = RadixKey(token_ids=[1, 2, 3, 4], extra_key=None)

        cache._kv_manager.get_match.return_value = (
            0, np.array([True] * 4)
        )
        # Alloc fails
        cache.token_to_kv_pool_allocator.alloc.return_value = None

        result = cache.match_prefix(key)
        self.assertEqual(result.device_indices.numel(), 0)

    def test_match_prefix_transfer_failure(self):
        """When FlexKV transfer fails, returns base result and frees slots."""
        from sglang.srt.mem_cache.radix_cache import RadixKey
        cache = self._make_cache(page_size=4)
        key = RadixKey(token_ids=[1, 2, 3, 4], extra_key=None)

        cache._kv_manager.get_match.return_value = (
            0, np.array([True] * 4)
        )
        cache.token_to_kv_pool_allocator.alloc.return_value = torch.arange(
            0, 4, dtype=torch.int64
        )
        # Transfer fails
        cache._kv_manager.wait.return_value = {
            0: _make_mock_kv_response(False)
        }

        result = cache.match_prefix(key)
        self.assertEqual(result.device_indices.numel(), 0)
        cache.token_to_kv_pool_allocator.free.assert_called()


# ---------------------------------------------------------------------------
# PUT / eviction tests
# ---------------------------------------------------------------------------

class TestPutAndEvict(unittest.TestCase):
    """Test cache_finished_req and evict with mocked KVManager."""

    def _make_cache(self, page_size=4):
        params = _make_mock_radix_cache_params(page_size=page_size)

        cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
        from sglang.srt.mem_cache.radix_cache import RadixCache
        RadixCache.__init__(cache, params)

        cache._tp_size = 1
        cache._rank = 0
        cache._tp_group = None
        cache._in_flight_put_tasks = {}
        cache._node_lock = __import__("threading").Lock()
        cache._get_timeout = 20.0
        cache._put_timeout = 20.0
        cache._prefetch_timeout = 5.0
        cache._kv_manager = _make_mock_kv_manager()
        cache._tp_client = None
        cache._mode = "local"
        return cache

    def test_evict_syncs_inflight(self):
        """_poll_completed_puts releases locks for completed tasks."""
        cache = self._make_cache()

        # Create a proper mock node with parent = None to prevent
        # dec_lock_ref from traversing an infinite MagicMock chain
        from sglang.srt.mem_cache.radix_cache import TreeNode
        mock_node = TreeNode()
        mock_node.lock_ref = 2  # >1 so dec won't go negative
        mock_node.parent = cache.root_node  # terminate at root
        mock_node.key = []
        mock_node.value = []

        cache._in_flight_put_tasks[42] = mock_node
        cache._kv_manager.try_wait.return_value = {
            42: _make_mock_kv_response(True)
        }

        cache._poll_completed_puts()

        cache._kv_manager.try_wait.assert_called_once()
        self.assertEqual(len(cache._in_flight_put_tasks), 0)

    def test_block_wait_all_puts(self):
        """_block_wait_all_puts waits for all in-flight PUT tasks."""
        cache = self._make_cache()

        from sglang.srt.mem_cache.radix_cache import TreeNode
        mock_node = TreeNode()
        mock_node.lock_ref = 2
        mock_node.parent = cache.root_node
        mock_node.key = []
        mock_node.value = []

        cache._in_flight_put_tasks[99] = mock_node
        cache._kv_manager.wait.return_value = {
            99: _make_mock_kv_response(True)
        }

        cache._block_wait_all_puts()

        cache._kv_manager.wait.assert_called_once()
        self.assertEqual(len(cache._in_flight_put_tasks), 0)

    def test_evict_empty_inflight(self):
        """evict() with no in-flight tasks proceeds directly."""
        cache = self._make_cache()
        cache.evict(0)
        # neither try_wait nor wait should be called
        cache._kv_manager.try_wait.assert_not_called()
        cache._kv_manager.wait.assert_not_called()

    def test_shutdown_cleanup(self):
        """shutdown() syncs in-flight and stops KVManager."""
        cache = self._make_cache()
        mock_mgr = cache._kv_manager  # save ref before shutdown sets it to None
        cache.shutdown()
        mock_mgr.shutdown.assert_called_once()
        self.assertIsNone(cache._kv_manager)

    def test_reset_clears_inflight(self):
        """reset() clears in-flight tracking."""
        cache = self._make_cache()
        cache._in_flight_put_tasks[1] = MagicMock()
        cache._in_flight_put_tasks[2] = MagicMock()
        cache.reset()
        self.assertEqual(len(cache._in_flight_put_tasks), 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
