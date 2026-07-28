# HIP Compute Kernel 设计文档与性能分析

> **已移除（历史文档）**：CE 内的 `COMPUTE_KERNEL` / `kernel_threshold` /
> `force_path=5` 路径已删除。CUDA-等价的 compute-kernel 传输请使用
> `use_ce_transfer=false`（`CLASSIC_KERNEL`，见 `csrc/transfer_kernels.cuh`）。
> 下文保留为当时 CE 内 CK 路径的性能分析记录。

## 概述

本方案实现了 HIP compute copy kernel，作为 SDMA (`hipMemcpyAsync`) 的替代路径，
用于 layerwise 小数据 KV cache 传输。经过实测分析和推理场景建模，**结论是
compute kernel 方案在实际推理部署中不应启用**，应保持 CE (SDMA) 作为默认传输
路径。

## 测试环境

- **GPU**: AMD Instinct MI308X (gfx942)
- **平台**: ROCm + PyTorch
- **传输模式**: MLA (kv_dim=1, chunk_size=1024B)
- **真实场景**: layer_granularity=1, sync=false, pinned metadata (模拟 LayerwiseTransferGroup)

## 正确性验证

`FLEXKV_TRANSFER_KERNEL_THRESHOLD=999999999` 强制所有传输走 compute kernel：

```
46 passed, 30 skipped, 0 failed (8.60s)
```

覆盖 MLA sharded/non-MLA × LAYERFIRST/BLOCKFIRST × layerwise H2D/D2H，全部通过。
功能实现本身是正确的。

## 性能数据（真实 layerwise: layer_granularity=1, async, pinned）

### H2D 对比

| Config | SDMA (us) | CK (us) | 加速比 | Auto (us, thr=24576) | Auto/SDMA |
|--------|:---:|:---:|:---:|:---:|:---:|
| 4 blk × 32 lyr (4KB/lyr) | 231 | 249 | 0.93x | 226 | 1.02x (选 SDMA) |
| 8 blk × 32 lyr (8KB/lyr) | 223 | 252 | 0.88x | 224 | 1.00x (选 SDMA) |
| 16 blk × 32 lyr (16KB/lyr) | 223 | 251 | 0.89x | 221 | 0.99x (选 SDMA) |
| **32 blk × 32 lyr (32KB/lyr)** | **339** | **261** | **1.30x** | 258 | **1.31x** (选 CK) |
| **64 blk × 32 lyr (64KB/lyr)** | **381** | **279** | **1.37x** | 285 | **1.34x** (选 CK) |

### D2H 对比（趋势一致）

| Config | SDMA (us) | CK (us) | 加速比 |
|--------|:---:|:---:|:---:|
| 4 blk × 32 lyr | 222 | 249 | 0.89x |
| 32 blk × 32 lyr | 321 | 248 | **1.30x** |
| 64 blk × 32 lyr | 337 | 248 | **1.36x** |

## 纯传输性能分析

### 交叉点在 ~24 blocks

SDMA 的 per-layer 开销 = `~2us × num_blocks`（线性增长，每次 `hipMemcpyAsync`
enqueue 一个 DMA 描述符），compute kernel 的 per-layer 开销 = `~7.8us`（常数，
即单次 kernel launch 开销）。交叉点在 `num_blocks ≈ 7.8 / 2 ≈ 24`。

- **num_blocks < 24**: SDMA 更快（O(N) 的常数项 < kernel launch 固定开销）
- **num_blocks >= 24**: Compute kernel 纯传输更快（O(N) 超过 O(1)）

### sync=true vs sync=false 行为差异

| 场景 | sync | SDMA 行为 | CK 优势 |
|------|:---:|-----------|---------|
| Whole-model | true | O(N) memcpy + 每次同步 | 3-7x（总是赢）|
| Layerwise | false | O(N) async memcpy（流水线）| 仅 num_blocks >= 24 时赢 |

sync=true 时 CK 总是赢，因为消除了 N 次同步开销。
sync=false 时 SDMA 的 async memcpy 流水线很高效。

## 推理场景分析（决定性因素）

纯传输性能只是一方面。关键在于实际推理中传输与计算的**资源竞争**关系，
这决定了 compute kernel 是否可用。

### D2H 场景：异步 offload，必须用 CE

D2H 是 offload 动作，推理引擎异步发起传输后**继续服务下一个推理请求**，
传输与计算并发执行：

```
时间线 ─────────────────────────────────────►
  [D2H 传输 KV cache (async)] ──────────────►
  [推理引擎: 服务下一个 request (计算)] ────►
              ↑ 两者并发
```

- **CE (SDMA)**：走专用 copy engine (DMA 引擎)，**不占 compute units (CUs)**，
  对计算 kernel 零干扰 → 传输和计算真正并行
- **Compute kernel**：占用 CUs，和正在跑的 attention/MLP 计算竞争资源 →
  **两者都变慢**，D2H offload 的意义被破坏

**结论**：D2H 必须用 CE，`kernel_threshold` 在 D2H 场景下有害。

### H2D 场景：layerwise overlap，同样受资源竞争影响

H2D 采用 layerwise 传输实现传输和计算 overlap，因为计算必须等到传输完成才能开始：

```
时间线 ──────────────────────────────────────────────►
  [H2D 传输 layer 0] → eventfd → [计算 layer 0 + H2D 传输 layer 1] → eventfd → [计算 layer 1 + ...]
                                        ↑ 这里 overlap
```

layer N 的计算和 layer N+1 的传输确实会 overlap：

- **CE 传输 layer N+1** → 不干扰 layer N 的计算 → overlap 有效
- **Compute kernel 传输 layer N+1** → 占用 CUs → **和 layer N 的计算冲突** →
  overlap 被破坏，实际端到端性能可能比纯传输时间更差

**结论**：即使实测中 compute kernel 的纯传输时间在大 block 数时更快 (1.3x)，
叠加与计算 kernel 的资源竞争后，实际端到端性能优势会消失甚至逆转。

### 实测数据印证

实际推理中单次 layerwise 传输通常是少量 blocks（几个到十几个 batch 的 prefill/decode），
正好落在 CE 占优的区间：

| 场景 | 典型 block 数 | 纯传输最优 | 考虑计算干扰后 |
|------|:---:|:---:|:---:|
| 小 batch decode | 4-16 | CE | CE（无争议）|
| 大 batch prefill | 32-64 | CK 快 1.3x | CE（CK 的优势被计算冲突抵消）|

## 最终结论

**compute kernel 方案在实际推理部署中不应启用**，两个方向都应保持使用 CE (SDMA)：

1. **D2H**：CE 走专用 copy engine，不干扰推理计算 ✓
2. **H2D layerwise**：CE 不与 overlap 的计算 kernel 竞争 CUs ✓

### 部署建议

- **`kernel_threshold` 默认值保持 0**（禁用 compute kernel 路径，保持现有 CE 行为）
- 代码保留作为 benchmark/debug 对比用途（`force_path=5` 可强制走 compute kernel
  做对比测试）
- 如未来在纯传输 benchmark 场景（无并发计算）下需要对比，可通过环境变量
  `FLEXKV_TRANSFER_KERNEL_THRESHOLD` 手动启用

## 实现保留说明

代码已完整实现并经过正确性验证，保留在代码库中：

- `ce_transfer.h`: `kernel_threshold` 字段、`COMPUTE_KERNEL=5` 枚举、函数声明
- `ce_transfer.cu`: 两个 HIP kernel（float4/int64）+ host launcher + pinned memory 检测
- `ce_transfer_dispatch.cu`: `total_bytes >= kernel_threshold` 时切换到 COMPUTE_KERNEL
- `bindings.cpp` + Python 层: `ce_kernel_threshold` 参数全链路接入
- `benchmarks/bench_layerwise_real.py`: 真实 layerwise 场景 benchmark

这些代码不会影响默认 CE 路径的性能（`kernel_threshold=0` 时不进入 compute kernel 分支），
仅作为可选的 benchmark 对比工具保留。
