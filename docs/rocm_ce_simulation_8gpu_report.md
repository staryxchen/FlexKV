# CE Transfer Monte Carlo Simulation Report — 8× MI308X (with COMPUTE_KERNEL path)

> **历史报告**：文中的 `ce_kernel` / `COMPUTE_KERNEL`（`force_path=5`）路径已从
> 代码中移除；请改用 classic CTA kernel（`use_ce=false` / `CLASSIC_KERNEL`）。

> **报告日期**: 2026-07-27
> **平台**: AMD Instinct MI308X × 8, ROCm 7.2.0, gfx942
> **工具**: `benchmarks/microbenchmark_ce_simulation.py`
> **配置**: 8 GPU, 30 rounds, 5 iters/round (median), seed=42
> **对比配置**: `baseline`(PER_BLOCK) / `opt`(choose_path 自适应) / `ce_kernel`(COMPUTE_KERNEL, force_path=5)
> **对照报告**: `docs/rocm_ce_simulation_report.md`（4 GPU, 仅 baseline/opt, `--skip-kernel`）
> **提交**: `57b068b` (feat/rocm-ce-backend)

---

## 1. 测试目标

在 8 张 AMD MI308X 上复现 CE 蒙特卡洛仿真，并**新增评估新实现的 HIP COMPUTE_KERNEL 路径**（提交 `e3df22b`），与既有 baseline/opt 配置三方对比，同时与 4 GPU 旧报告数据对照。

### 新增配置说明

| 配置 | use_ce | path_opt | force_path | 含义 |
|------|--------|----------|------------|------|
| `baseline` | True | False | -1 | CE PER_BLOCK（逐 block 循环 memcpy） |
| `opt` | True | True | -1 | CE choose_path 自适应（CONTIG/SEGMENT/GATHER 自动选择） |
| `ce_kernel` | True | True | 5 | **CE COMPUTE_KERNEL**（HIP compute-copy kernel，新实现） |

> **重要**：旧的 `kernel` 配置（`use_ce=False`，自定义 CUDA kernel）在 ROCm 构建中被显式禁止（`ce_transfer_dispatch.hip:27`），本次运行使用 `--skip-kernel` 跳过。新增的 `ce_kernel` 是 CE dispatch 内部的 `CEPath::COMPUTE_KERNEL`（force_path=5），通过 `ce_force_path` 参数接入 benchmark。

### 测试矩阵

| 维度 | 取值 |
|------|------|
| 数据规模 | `small` (32 layers, pool=512), `medium` (61, 2048), `large` (80, 8192) |
| 内存布局 | `lfirst` (layer-first), `bfirst` (block-first) |
| 模型配置 | MHA, MLA-sharded, MLA-rank0_only, MLA-layer_parallel, MLA-rank_rotate |
| 碎片化 | 30 组固定 `(batch_frac, target_seg)` 对，seed=42 |
| GPU 数 | **8**（旧报告为 4） |

> **数据量差异**：MHA 模式 `num_head=num_gpus`，8 GPU 下每 block 数据量为 4 GPU 的 2 倍；MLA 模式 `num_head=1`，数据量不随 GPU 数变化。

---

## 2. 总体结果

### 2.1 三方加速比分布（vs baseline，RT = D2H + H2D）

| 统计项 | opt | ce_kernel |
|--------|-----|-----------|
| 总数据点 | 30 | 30 |
| 中位加速比 | **37.3x** | **109.4x** |
| 最小加速比 | 13.8x | 74.5x |
| 最大加速比 | 102.3x | 214.2x |
| P25 | 27.7x | 88.7x |
| P75 | 62.5x | 128.0x |

### 2.2 ce_kernel vs opt（纯传输场景）

| 统计项 | 值 |
|--------|-----|
| ce_kernel 快于 opt 的组合数 | **30/30 (100%)** |
| 中位倍率 (opt_RT / ce_kernel_RT) | **2.65x** |
| 最小倍率 | 1.15x (large bfirst MLA-rank0_only) |
| 最大倍率 | 13.34x (small lfirst MHA) |

### 2.3 核心结论

- **ce_kernel 在纯传输场景下全面优于 opt**：30 个组合中 100% 胜出，中位快 2.65x
- **ce_kernel vs baseline 中位 109.4x 加速**，最差场景仍有 74.5x（远优于 opt 的最差 13.8x）
- **opt vs baseline 中位 37.3x**，相比 4 GPU 旧报告（中位 29.9x）有提升
- ⚠️ **关键限制**：ce_kernel 使用 compute unit 执行 copy，真实推理中会与 attention/MLP 计算竞争 CU，破坏传输/计算重叠。本次为**纯传输基准**，生产环境仍推荐 `opt`（CE/SDMA 专用 copy engine）。详见提交 `e3df22b` 及 `docs/hip_compute_kernel_perf.md`。

---

## 3. 与 4 GPU 旧报告对照（opt vs baseline RT 加速比）

| 组合 | 4 GPU opt RT spd | 8 GPU opt RT spd | 变化 |
|------|:-:|:-:|:-:|
| large bfirst MHA | 41x | **60.0x** | ↑ |
| large bfirst MLA-layer_parallel | 44x | **62.5x** | ↑ |
| large bfirst MLA-rank0_only | 108x | 102.3x | ↓ |
| large bfirst MLA-rank_rotate | 114x | 74.5x | ↓ |
| large bfirst MLA-sharded | 57x | **91.6x** | ↑ |
| large lfirst MHA | 13x | **17.9x** | ↑ |
| large lfirst MLA-layer_parallel | 44x | **62.6x** | ↑ |
| large lfirst MLA-rank0_only | 42x | **65.4x** | ↑ |
| large lfirst MLA-rank_rotate | 46x | **54.4x** | ↑ |
| large lfirst MLA-sharded | 40x | **85.5x** | ↑ |
| medium bfirst MHA | 30x | **39.4x** | ↑ |
| medium bfirst MLA-layer_parallel | 28x | **32.7x** | ↑ |
| medium bfirst MLA-rank0_only | 65x | 61.8x | ↓ |
| medium bfirst MLA-rank_rotate | 62x | 36.5x | ↓ |
| medium bfirst MLA-sharded | 35x | **58.1x** | ↑ |
| medium lfirst MHA | 13x | **14.1x** | ↑ |
| medium lfirst MLA-layer_parallel | 22x | **39.0x** | ↑ |
| medium lfirst MLA-rank0_only | 23x | **38.0x** | ↑ |
| medium lfirst MLA-rank_rotate | 20x | **29.4x** | ↑ |
| medium lfirst MLA-sharded | 27x | **45.4x** | ↑ |
| small bfirst MHA | 26x | **27.1x** | ↑ |
| small bfirst MLA-layer_parallel | 18x | **25.4x** | ↑ |
| small bfirst MLA-rank0_only | 24x | **32.2x** | ↑ |
| small bfirst MLA-rank_rotate | 23x | 20.7x | ↓ |
| small bfirst MLA-sharded | 25x | **28.3x** | ↑ |
| small lfirst MHA | 10x | **16.0x** | ↑ |
| small lfirst MLA-layer_parallel | 21x | 17.7x | ↓ |
| small lfirst MLA-rank0_only | 21x | 19.0x | ↓ |
| small lfirst MLA-rank_rotate | 21x | 13.8x | ↓ |
| small lfirst MLA-sharded | 24x | 21.1x | ↓ |

> **趋势**：22/30 组合的 opt 加速比在 8 GPU 下提升，主要因 MHA 数据量翻倍放大了 PER_BLOCK 循环开销。MLA 场景（数据量不变）加速比基本持平或略降，符合预期。少数 rank_rotate/rank0_only 组合下降，可能因 8 GPU 下 GPU 同步开销增加以及随机碎片化分布差异（8 GPU batch 更大导致段分布变化）。

---

## 4. 完整明细表（8 GPU, 3 配置）

| Size | Layout | Model | Baseline RT (ms) | Opt RT (ms) | CK RT (ms) | Opt spd | CK spd | CK/Opt |
|------|--------|-------|:-:|:-:|:-:|:-:|:-:|:-:|
| large | bfirst | MHA | 2916.7 | 48.6 | 25.7 | 60.0x | 113.7x | 1.9x |
| large | bfirst | MLA-layer_parallel | 754.7 | 12.1 | 7.5 | 62.5x | 100.4x | 1.6x |
| large | bfirst | MLA-rank0_only | 1083.1 | 10.6 | 9.2 | 102.3x | 117.2x | 1.1x |
| large | bfirst | MLA-rank_rotate | 992.7 | 13.3 | 9.4 | 74.5x | 105.1x | 1.4x |
| large | bfirst | MLA-sharded | 1402.3 | 15.3 | 7.9 | 91.6x | 177.6x | 1.9x |
| large | lfirst | MHA | 2849.0 | 159.4 | 24.7 | 17.9x | 115.4x | 6.5x |
| large | lfirst | MLA-layer_parallel | 726.8 | 11.6 | 7.6 | 62.6x | 95.8x | 1.5x |
| large | lfirst | MLA-rank0_only | 1088.6 | 16.7 | 9.3 | 65.4x | 117.0x | 1.8x |
| large | lfirst | MLA-rank_rotate | 1070.9 | 19.7 | 9.4 | 54.4x | 114.1x | 2.1x |
| large | lfirst | MLA-sharded | 1418.1 | 16.6 | 7.8 | 85.5x | 180.8x | 2.1x |
| medium | bfirst | MHA | 488.2 | 12.4 | 5.1 | 39.4x | 96.3x | 2.4x |
| medium | bfirst | MLA-layer_parallel | 123.7 | 3.8 | 1.6 | 32.7x | 75.3x | 2.3x |
| medium | bfirst | MLA-rank0_only | 188.6 | 3.1 | 2.0 | 61.8x | 96.5x | 1.6x |
| medium | bfirst | MLA-rank_rotate | 182.7 | 5.0 | 2.2 | 36.5x | 85.0x | 2.3x |
| medium | bfirst | MLA-sharded | 253.6 | 4.4 | 1.7 | 58.1x | 150.7x | 2.6x |
| medium | lfirst | MHA | 426.4 | 30.3 | 4.8 | 14.1x | 88.4x | 6.3x |
| medium | lfirst | MLA-layer_parallel | 126.5 | 3.2 | 1.6 | 39.0x | 80.6x | 2.1x |
| medium | lfirst | MLA-rank0_only | 194.3 | 5.1 | 1.9 | 38.0x | 102.3x | 2.7x |
| medium | lfirst | MLA-rank_rotate | 187.1 | 6.4 | 2.0 | 29.4x | 93.4x | 3.2x |
| medium | lfirst | MLA-sharded | 219.9 | 4.8 | 1.7 | 45.4x | 128.3x | 2.8x |
| small | bfirst | MHA | 52.8 | 1.9 | 0.3 | 27.1x | 192.2x | 7.1x |
| small | bfirst | MLA-layer_parallel | 15.2 | 0.6 | 0.2 | 25.4x | 74.5x | 2.9x |
| small | bfirst | MLA-rank0_only | 23.5 | 0.7 | 0.2 | 32.2x | 116.2x | 3.6x |
| small | bfirst | MLA-rank_rotate | 22.2 | 1.1 | 0.3 | 20.7x | 79.3x | 3.8x |
| small | bfirst | MLA-sharded | 27.0 | 1.0 | 0.2 | 28.3x | 131.9x | 4.7x |
| small | lfirst | MHA | 53.7 | 3.3 | 0.3 | 16.0x | 214.2x | 13.3x |
| small | lfirst | MLA-layer_parallel | 15.1 | 0.9 | 0.2 | 17.7x | 83.6x | 4.7x |
| small | lfirst | MLA-rank0_only | 23.1 | 1.2 | 0.2 | 19.0x | 121.2x | 6.4x |
| small | lfirst | MLA-rank_rotate | 22.3 | 1.6 | 0.3 | 13.8x | 82.5x | 6.0x |
| small | lfirst | MLA-sharded | 26.1 | 1.2 | 0.2 | 21.1x | 133.6x | 6.3x |

> spd = baseline / config（越高越快）；CK/Opt = opt_RT / ce_kernel_RT（越高说明 ce_kernel 相对 opt 越快）

---

## 5. 分维度分析

### 5.1 按数据规模（中位 RT 加速比）

| 规模 | opt vs baseline | ce_kernel vs baseline | ce_kernel vs opt |
|------|:-:|:-:|:-:|
| small | 21.1x | 121.2x | 5.0x |
| medium | 38.0x | 93.4x | 2.4x |
| large | 65.4x | 114.1x | 1.8x |

> **趋势**：
> - opt 加速比随数据量增大而提升（PER_BLOCK 循环开销被放大）
> - ce_kernel 加速比在 small 时最高（214x），因小数据量下 compute kernel 启动开销远低于 SDMA 多段 memcpy 开销
> - **ce_kernel vs opt 优势随数据量增大而缩小**（small 5.0x → large 1.8x），因大数据量下 SDMA 吞吐优势发挥，差距收敛

### 5.2 按内存布局（中位 RT 加速比）

| 布局 | opt vs baseline | ce_kernel vs baseline | ce_kernel vs opt |
|------|:-:|:-:|:-:|
| lfirst | 29.4x | 102.3x | 3.2x |
| bfirst | 54.4x | 105.1x | 2.2x |

> **趋势**：ce_kernel 对布局不敏感（102x vs 105x），因 compute kernel 直接按 block_id 索引复制，物理连续性不影响性能。而 opt 在 bfirst 下加速更高（54x vs 29x），因 bfirst 物理连续性好，CONTIG_DIRECT/SEGMENT_DIRECT 命中率高。

### 5.3 按模型配置（中位 RT 加速比）

| 模型配置 | opt vs baseline | ce_kernel vs baseline | ce_kernel vs opt |
|---------|:-:|:-:|:-:|
| MHA | 17.9x | 115.4x | 6.5x |
| MLA-sharded | 58.1x | 150.7x | 2.6x |
| MLA-layer_parallel | 39.0x | 80.6x | 2.1x |
| MLA-rank0_only | 38.0x | 117.0x | 2.7x |
| MLA-rank_rotate | 29.4x | 93.4x | 3.2x |

> **关键发现**：MHA 场景下 ce_kernel vs opt 优势最大（6.5x），因 MHA（kv_dim=2）数据量大，opt 在 lfirst 下走 GATHER 路径开销高（opt RT 159ms vs ce_kernel 25ms for large lfirst MHA），而 ce_kernel 直接 compute copy 不受碎片化影响。

### 5.4 ce_kernel H2D 时间稳定性

ce_kernel 的 H2D 时间在不同 MLA mode 下高度一致（large: ~6.4ms, medium: ~1.3ms, small: ~0.12ms），仅 MHA 翻倍（kv_dim=2）。这证实 compute kernel 按 data size 线性复制，与 block-id 碎片化分布无关。

---

## 6. 关键发现

### 6.1 ce_kernel 纯传输性能卓越

| 场景 | Baseline RT | Opt RT | CK RT | CK 优势 |
|------|:-:|:-:|:-:|:-:|
| small lfirst MHA | 53.7ms | 3.3ms | 0.25ms | 13.3x vs opt |
| large lfirst MHA | 2849ms | 159ms | 24.7ms | 6.5x vs opt |
| large bfirst MLA-sharded | 1402ms | 15.3ms | 7.9ms | 1.9x vs opt |
| large bfirst MLA-rank0_only | 1083ms | 10.6ms | 9.2ms | 1.1x vs opt |

> ce_kernel 在 MHA + lfirst（opt 最弱场景）优势最大，因 compute kernel 绕过了 GATHER_SCATTER 路径的高开销。

### 6.2 opt 的最差场景被 ce_kernel 大幅改善

旧报告中 opt 最差场景为 MHA + lfirst（13-18x）。8 GPU 下：
- small lfirst MHA: opt 16.0x → ce_kernel **214.2x**
- medium lfirst MHA: opt 14.1x → ce_kernel **88.4x**
- large lfirst MHA: opt 17.9x → ce_kernel **115.4x**

### 6.3 生产环境的关键权衡

| 因素 | opt (CE/SDMA) | ce_kernel (COMPUTE_KERNEL) |
|------|---------------|---------------------------|
| 纯传输延迟 | 较高（受碎片化影响） | **极低**（不受碎片化影响） |
| 真实推理重叠 | **优**（专用 copy engine，不竞争 CU） | 差（竞争 attention/MLP 的 CU） |
| 大数据量吞吐 | **优**（SDMA 带宽优势） | 良（受 CU 数限制） |
| 小数据量延迟 | 较高（SDMA 启动开销） | **优**（kernel 启动快） |
| 生产推荐 | ✅ **推荐** | ❌ 仅 benchmark/debug |

> 根据 `e3df22b` 提交说明：真实推理中 ce_kernel 与 attention/MLP compute kernel 竞争 CU，破坏传输/计算重叠。CE (SDMA, 专用 copy engine) 在 D2H（异步 offload）和 H2D（layerwise）两个方向上都是正确选择。

---

## 7. 结论与建议

### 7.1 验证结论

1. **opt (CE choose_path) 在 8 GPU 上验证成熟**：30 组合中位 37.3x 加速，相比 4 GPU（29.9x）提升，22/30 组合优于 4 GPU
2. **ce_kernel (COMPUTE_KERNEL) 纯传输性能全面领先**：100% 组合优于 opt，中位 2.65x，最差仍 1.15x
3. **ce_kernel 消除了 opt 的最差场景**：MHA + lfirst 从 13-18x 提升至 88-214x
4. **ce_kernel 对碎片化/布局完全免疫**：H2D 时间仅取决于数据量，与 block-id 分布无关

### 7.2 部署建议

| 场景 | 推荐配置 | 理由 |
|------|---------|------|
| 生产推理（有计算重叠） | `opt` (path_opt=on) | SDMA 专用 copy engine，不竞争 CU |
| 纯传输/benchmark/debug | `ce_kernel` (force_path=5) | 纯传输延迟最低 |
| 小数据量 layerwise 传输 | `opt` + 考虑 kernel_threshold | 可设 `FLEXKV_TRANSFER_KERNEL_THRESHOLD` 在小 block 时走 CK 降低延迟，大 block 走 CE 保吞吐 |
| MHA + lfirst 场景 | `opt`（但考虑 CK 作为 fallback） | opt 此场景最弱（17.9x），CK 可达 115x |

### 7.3 待优化方向

1. **混合策略**：探索根据数据量自适应切换 CK/CE —— 小数据量用 CK（低延迟），大数据量用 CE（高吞吐 + 不竞争 CU）
2. **MHA + lfirst 的 opt 路径**：仍是最弱场景（17.9x），可优化 GATHER kernel 向量化
3. **8 GPU 同步开销**：部分 MLA rank_rotate 组合加速比下降，需排查 8 GPU 同步开销

---

## 附录 A: 测试配置

- **随机种子**: 42 (可复现，与 4 GPU 报告一致)
- **碎片化参数**: 30 组预生成 `(batch_frac, target_seg)` 对
- **segment_threshold**: 8
- **path_opt**: baseline=False, opt=True, ce_kernel=True(force_path=5)
- **memcpy2d**: off
- **transfer_num_cta**: 16
- **iters/round**: 5 (median), warmup=3
- **block 池规模**: small=512, medium=2048, large=8192
- **batch 分布**: small(27-489), medium(108-1959), large(435-7836)

## 附录 B: 代码变更

本次 benchmark 新增 `ce_kernel` 配置（`benchmarks/microbenchmark_ce_simulation.py`）：
- `CE_CONFIGS` 改为 4 元组 `(label, use_ce, path_opt, force_path)`，新增 `("ce_kernel", True, True, 5)`
- `make_tp_group` 新增 `ce_force_path` 参数，透传至 `TPTransferThreadGroup`
- 运行时 `--skip-kernel` 跳过旧的 `use_ce=False` kernel（ROCm 禁用），保留新的 `ce_kernel`
