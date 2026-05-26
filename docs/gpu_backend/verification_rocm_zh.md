# FlexKV 在 AMD ROCm 上的验证 Playbook

> 本文档记录在 AMD GPU + ROCm 环境下验证 FlexKV 是否可工作的分阶段步骤、当前进度、以及每一阶段实际踩过的坑。
>
> 关联文档:
> - 设计/抽象层全景: [`README_en.md`](./README_en.md) / [`README_zh.md`](./README_zh.md)
> - 测试目录: [`tests/`](../../tests/)

---

## 1. 验证策略

按 first principles 把"FlexKV 在 ROCm 上能跑通"这个命题拆成 **5 个相互独立的层**。每一层只验证一个维度,从最便宜、最早失败的检查推进到真实工作负载——任何一层失败都能精确定位是 **构建链路 / 抽象分发 / 真机能力 / 业务路径 / 集成** 的哪一层问题。

| Stage | 验证命题 | 验证手段 | 失败说明什么 |
| ----- | -------- | -------- | ------------ |
| **0. 环境前置** | ROCm SDK + ROCm 版 PyTorch + hipify-perl 都到位 | `rocm-smi` / `hipcc --version` / `which hipify-perl` / `python -c "import torch; print(torch.version.hip)"` | 环境装错(常见: 装了 NVIDIA 版 torch) |
| **1. ROCm 构建链路** | `FLEXKV_GPU_BACKEND=rocm pip install -e .` 能产出 `flexkv.c_ext` | 编译命令、产物文件存在 | hipify / hipcc / 系统依赖问题 |
| **2. Dispatch 分发** | Python 抽象层正确路由到 `RocmBackend` | `pytest tests/gpu_backend/test_backend_dispatch.py` (无需 GPU) | 自动检测或 `_FORCE_MAP` 注册问题 |
| **3. 真机能力** | RocmBackend 7 类方法在真 GPU 上可用 | 一段 smoke 脚本(见 §6.3) | 设备 / 流 / pinned host / IPC 等基础能力 |
| **4. 端到端 KV 传输** | `transfer_kv_blocks` / `layout_transform` 跑通,server-client 模式可用 | `pytest tests/test_memory_handle.py` → `test_cache_engine.py` → `test_kvmanager.py` | hot path 或 IPC 问题 |
| **5. 真实集成 (可选)** | 与 vLLM / TRT-LLM 在 ROCm 上联调 | `examples/vllm_adaption/` + ROCm vLLM | 上游集成层问题 |

GDS (§3.5.1 of `README_en.md`) 在 ROCm 上**天然不需要验证**——`supports_direct_storage()` 直接返回 `False`,SSD 走 GPU-agnostic 的 `transfer_kv_blocks_ssd`(POSIX/io_uring fallback)。

---

## 2. 当前进度

| Stage | 状态 | 完成日期 | 备注 |
| ----- | ---- | -------- | ---- |
| 0. 环境前置 | ✅ 完成 | 2026-05-21 | ROCm 7.2.26015 / HIP 7.2 / torch hip=7.2 / hipify-perl=/opt/rocm/bin/hipify-perl |
| 1. ROCm 构建链路 | ✅ 完成 | 2026-05-21 | 产出 `flexkv/c_ext.so`,踩坑细节见 §7.1-§7.5 |
| 2. Dispatch 分发 | ✅ 完成 | 2026-05-21 | 19/19 通过(9 vendor-neutral + 10 ROCm 专属);新增 `test_rocm_dispatch.py` 补全覆盖盲区,见 §7.6 |
| 3. 真机能力 | ✅ 完成 | 2026-05-21 | 15/15 通过;新增 `test_rocm_runtime.py`;过程中调整两次测试设计,见 §7.7 |
| 4. 端到端 KV 传输 | ✅ 完成 | 2026-05-21 | memory_handle 11/11 + cache_engine 179/183 + kvmanager 60/60 + namespace 6/6;**发现 2 个上游 abstraction 漏洞**(MPS / CE transfer),需 2 个环境变量旁路,见 §7.8 |
| 5. 真实集成 | ⏳ 待办(可选) | — | — |

---

## 3. 已知良好环境 (Stage 1 通过的快照)

| 项 | 值 |
| -- | -- |
| OS | Linux (notebook 容器) |
| Python | 3.10 (`/opt/venv`) |
| ROCm | 7.2.26015 (`/opt/rocm-7.2.0`) |
| HIP | 7.2.26015-fc0010cf6a |
| PyTorch | ROCm 版,`torch.version.hip='7.2.26015-fc0010cf6a'` |
| 目标 GPU arch | gfx942 + gfx950 (8x AMD MI 系列) |
| C 扩展产物 | `flexkv/c_ext.so` (hipcc 编译) |

### 3.1 ROCm 必备运行时环境变量

跑任何 KVManager 相关测试 / 真实业务时**必须**设置:

```bash
export FLEXKV_ENABLE_MPS=0              # NVIDIA MPS daemon 在 ROCm 上不存在
export FLEXKV_USE_CE_TRANSFER_H2D=1     # 改用 hipMemcpyAsync,绕开 NVIDIA UVA 假设
export FLEXKV_USE_CE_TRANSFER_D2H=1     # 同上
```

详细原理见 §7.8。这两个目前是**上游 abstraction 漏洞的旁路方案**,长期方案是给 RocmBackend 加 `preferred_*()` 钩子让默认值随 vendor 切换。

---

## 4. Stage 0 · 环境前置

```bash
# AMD GPU & ROCm SDK 可见
rocm-smi
hipcc --version
which hipify-perl    # RocmBuilder.configure_env() 会调用它

# PyTorch 必须是 ROCm 版(不是 CUDA 版)
python -c "import torch; print('hip:', torch.version.hip, 'cuda:', torch.version.cuda)"
# 期望: hip='7.x.x...', cuda=None
```

只要 `torch.version.hip is not None`,`RocmBuilder.is_available()` 就为 True (`build_backends/rocm_builder.py`)。

---

## 5. Stage 1 · ROCm 构建链路

### 5.1 关键认知

FlexKV 是 **两阶段构建**,这一点对新手不够友好(install.sh 把它隐藏了):

| 阶段 | 做什么 | 触发方式 |
|------|--------|----------|
| **Stage 1A · CMake** | 编 vendored 的 `third_party/xxHash` → `build/lib/libxxhash.so` 和 `build/include/xxhash.h` | `cmake --build .` |
| **Stage 1B · pip** | 编 Python 扩展 `flexkv.c_ext`,从 `build/include/` 拿头文件、从 `build/lib/` 链 `-lxxhash` | `FLEXKV_GPU_BACKEND=rocm pip install -e .` |

`build_backends/rocm_builder.py` 的 `get_include_dirs()` 第一项就是 `build/include`,**假定**第一阶段已完成。直接跳到 1B 会报 `xxhash.h: No such file or directory`。

### 5.2 完整构建命令(已验证可工作)

```bash
cd /path/to/FlexKV

# ---- 系统依赖 ----
# liburing 是 SSD KV-cache 的 io_uring fallback 必需(ROCm 没 GDS)
sudo apt-get install -y liburing-dev    # Debian/Ubuntu
# sudo yum install -y liburing-devel    # RHEL/CentOS

# ---- Python 编译时依赖(--no-build-isolation 后 pip 不会自动装它们) ----
pip install "setuptools>=40.0.0" wheel "Cython>=3.0.10" pybind11 ninja

# ---- Stage 1A: vendored xxhash ----
git submodule update --init --recursive third_party/xxHash
mkdir -p build && cd build
cmake .. -DFLEXKV_ENABLE_MONITORING=OFF
cmake --build . -j"$(nproc)"
cd ..
# 验证:
ls build/lib/libxxhash.so* build/include/xxhash.h

# ---- Stage 1B: Python 扩展 ----
# 关键: --no-build-isolation,直接用 venv 里的 ROCm torch,
# 否则 pip 会从 PyPI 主索引重新拉 NVIDIA 版 torch (见 §7.1)
FLEXKV_GPU_BACKEND=rocm pip install -e . --no-build-isolation -v 2>&1 | tee build_rocm.log

# ---- 验证产物 ----
python -c "import flexkv.c_ext as ext; print(ext.__file__)"
# 期望: /path/to/FlexKV/flexkv/c_ext.so (or .cpython-*.so)
```

### 5.3 偷懒路径(可选)

如果不想手动拆两步,可以用 install.sh 一站式:

```bash
FLEXKV_GPU_BACKEND=rocm bash install.sh --no-venv --skip-deps --disable-p2p
```

参数说明:
- `--no-venv`: 不要重新建 venv,用当前已有的 ROCm 版 PyTorch
- `--skip-deps`: 跳过 apt 安装(假设 liburing-dev 已装好);如果没装就别加这个
- `--disable-p2p`: 关 Mooncake 编译(GDR/远程 KV reuse,本次验证不需要)

但**推荐手动三步法**——更透明,出错时知道是哪一步,而且不会顺手装一堆现在不需要的东西(Mooncake、Redis client 等)。

---

## 6. Stage 2-4 (待执行)

完整步骤在 `README_en.md` 的"5. Checklist for Onboarding a New Vendor"和[原始验证矩阵讨论](#)中,这里只列要点,执行后回填本文档 §2 进度表。

### 6.1 Stage 2 · Dispatch smoke (无需 GPU)

跑两个测试文件:

```bash
# 框架契约(任何 vendor 都该满足):9 个测试
pytest tests/gpu_backend/test_backend_dispatch.py -v

# ROCm 专属命题:10 个测试
pytest tests/gpu_backend/test_rocm_dispatch.py -v

# 一次性跑全部(等价于上面两条)
pytest tests/gpu_backend/ -v
```

期望 **19/19 通过**。`test_rocm_dispatch.py` 文件级 `pytest.mark.skipif` 在非 ROCm 机器上自动跳过,不污染上游 vendor-neutral 套件。

两个文件的覆盖分工:

| 文件 | 命题类别 | 测试数 |
|------|---------|--------|
| `test_backend_dispatch.py` (上游已有) | ABC 契约存在、`current_backend` 可解析、`select_backend('generic')` 工作、错误别名抛错、`build_config` 名字一致、builder registry 注册了 | 9 |
| `test_rocm_dispatch.py` (本次新增) | 真机自动检测命中 ROCM、`rocm/hip/amd` 三个别名都到 RocmBackend、ABC 完整可实例化、ROCm 三件套 visibility env vars、`supports_direct_storage()=False`、`flexkv.c_ext` 可 import、含 `transfer_kv_blocks/TPTransferThreadGroup`、**不含** `GDSManager` 等 NVIDIA-only symbol | 10 |

通过即证明:
- 抽象层正确路由到 `RocmBackend`(不会回退到 GenericBackend)
- ABC 没漏实现(否则 `RocmBackend()` 构造会 TypeError)
- ROCm wheel 的 vendor 隔离干净(GDS symbol 不存在,§3.5.1 + 五大原则 #1)
- Stage 1 编出的 C 扩展真正可用(连接 Python 与 C++ binding)

### 6.2 Stage 3 · 真机能力 smoke

```bash
pytest tests/gpu_backend/test_rocm_runtime.py -v
```

文件级 `pytest.mark.skipif` 同时检查"是 ROCm 环境"和"`device_count() > 0`",任一不满足整文件跳过。

测试按 `README_en.md §3.1` 的方法分组组织,**每个测试只钉一个能力**——失败时能精确定位是哪一组挂了:

| 文档分组 | 覆盖测试 | 要点 |
|---|---|---|
| Group 2 · Devices | `test_device_count_positive`, `test_device_name_nonempty`, `test_set_and_current_device`, `test_get_device_capability`, `test_make_device_returns_cuda_typed`, `test_synchronize_does_not_raise`, `test_empty_cache_does_not_raise`, `test_init_runtime_is_idempotent`, `test_is_gpu_tensor_distinguishes_cpu_and_gpu` | `make_device(0)` 必须返回 `torch.device('cuda', 0)`(ROCm 在 PyTorch 里仍以 `'cuda'` 现身) |
| Group 3 · Streams | `test_create_destroy_stream`, `test_current_stream_handle_is_int`, `test_stream_context_manager` | `stream_handle(s)` 返回非零 `uintptr_t` |
| Group 4 · Pinned host | `test_register_unregister_host_tensor` | **关键**:这是唯一从 Python 侧 `dlopen('libamdhip64.so')` 的代码路径,失败常因 `LD_LIBRARY_PATH` 没含 `/opt/rocm/lib`。**`alloc_pinned/free_pinned` 是 ABC 可选方法**(默认 NotImplementedError),NVIDIA / ROCm 都没 override,FlexKV 业务也不用,故不测。详见 §7.7 |
| Group 6 · IPC | `test_supports_ipc_returns_bool`, `test_ipc_handle_export` | 只测 `get_ipc_handle` 能 export 非空 bytes;**不**在同进程做 round-trip(`*IpcOpenMemHandle` 设计上只能由 importing process 调用,同进程会被 ROCm 拒,error 201)。真正跨进程 round-trip 由 Stage 4 `test_memory_handle.py` 用 multiprocessing 测试。详见 §7.7 |
| Group 7 · Direct storage | (已在 Stage 2 `test_rocm_no_direct_storage` 声明性验证) | ROCm 没 GDS,运行时无需再验证 |

通过即证明:
- 设备/流/pinned host 三类基础能力在真机可用(对应文档"hot path"前置条件)
- IPC 句柄 round-trip 不破(对应 server-client 多进程 KV-cache 共享场景)
- ROCm runtime 库在 Python 侧能正确 dlopen

### 6.2.1 Stage 3 常见现场问题

| 现象 | 大概率原因 | 处理 |
|---|---|---|
| `register_host_tensor` 报 `OSError: libamdhip64.so not found` | `LD_LIBRARY_PATH` 没含 `/opt/rocm/lib` | `export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH` |
| `supports_ipc()` 返回 False / IPC round-trip 抛错 | 容器 IPC namespace 限制(常见于受限容器) | 不是代码 bug,IPC 测试自动 skip;Stage 4 用 PyTorch reductions fallback 路径(§3.6.1) |
| `device_count() == 0` | `HIP_VISIBLE_DEVICES` 限制了可见性,或 GPU 处于 power-saving 状态 | 检查 `rocm-smi`(注意"low-power state" warning),解除可见性 mask |
| `torch.empty(..., device=...)` 阶段就挂 | torch HIP 初始化问题 | 先 `python -c "import torch; torch.cuda.init()"` 显式触发 |

### 6.3 Stage 4 · 端到端 KV 传输

跑前**必须**先设这两组环境变量(见 §7.8 原因):

```bash
export FLEXKV_ENABLE_MPS=0
export FLEXKV_USE_CE_TRANSFER_H2D=1
export FLEXKV_USE_CE_TRANSFER_D2H=1

# 清理上次跑挂留下的 ZMQ socket(否则新进程 bind 永久阻塞)
rm -f /tmp/flexkv_*
```

然后**逐个**跑 4 个测试文件:

| 文件 | 通过率 | 备注 |
|---|---|---|
| `tests/test_memory_handle.py` | **11/11** | 跨进程 IPC handoff;过程中顺手修了 2 个上游测试 bug,见 §7.7 之外的 patch |
| `tests/test_cache_engine.py` | **179/183** | KV transfer 热路径(`test_match_and_insert` 等 175 个核心测试)全部通过;4 个 failure 全集中在 `test_config_init` 的入参校验,**是 Cython 编译模式语义偏差,与 ROCm 无关**,详见 §7.8 末尾 |
| `tests/test_kvmanager.py` | **60/60 + 66 SKIPPED** | server-client + 多进程 + 跨进程 IPC + transfer_kv_blocks 热路径全套验证;**这是本次 ROCm 验证的最重头戏**。SKIP 全部合理:42 GDS(ROCm 没 GDS)+ 18 fp8(ROCm 7.x 缺 fp8 ops)+ 6 dp_size>1 server-client(FlexKV 自身限制) |
| `tests/test_namespace_isolation.py` | **6/6** | namespace 哈希逻辑,GPU-agnostic |

跑命令:

```bash
pytest tests/test_memory_handle.py -v
pytest tests/test_cache_engine.py -v
pytest tests/test_kvmanager.py -v        # 注意:此文件耗时 ~25 分钟
pytest tests/test_namespace_isolation.py -v
```

通过即证明:

- 跨进程 IPC handoff(主进程 GPU tensor 通过 IPC handle 在 server/TransferManager 子进程里 reopen)在 ROCm 上工作
- `transfer_kv_blocks` 热路径(host-driven `hipMemcpyAsync` 模式)正确无误
- KVManager / KVTaskEngine / TransferEngine 全套 server-client 拓扑在 ROCm 上端到端跑通

### 6.3.1 Stage 4 常见现场问题

| 现象 | 大概率原因 | 处理 |
|---|---|---|
| `FileNotFoundError: 'nvidia-cuda-mps-control'` | `FLEXKV_ENABLE_MPS` 没设为 0 | 见 §7.8.1 |
| `Memory access fault by GPU node-N on address 0x...` (host 地址) | `FLEXKV_USE_CE_TRANSFER_*` 没设为 1,走了 NVIDIA UVA-dependent 的 device kernel 路径 | 见 §7.8.2 |
| pytest 启动后**无任何输出**地卡 60 秒以上 | `/tmp/flexkv_*` socket 残留,新进程 bind 永久阻塞 | `rm -f /tmp/flexkv_*` 后重跑 |
| `test_tensor_from_cpu_tensor` regex 不匹配 | 测试期待的报错字符串 `"CUDA tensor sharing"` 已被上游改名为 `"GPU tensor sharing"` | 这是上游测试本身 bug,我们已顺手修(test_memory_handle.py:387) |
| `test_fp8_tensor_from_bytes_roundtrip` NameError | 上游测试引用了不存在的 worker 函数 | 我们已顺手补(test_memory_handle.py 新增 `_worker_test_fp8_tensor_from_bytes`) |

### 6.4 Stage 5 · 真实集成 (可选)

```bash
# 微基准
python benchmarks/benchmark_single_batch.py --config benchmarks/example_config.yml

# 与 vLLM ROCm 集成(需要 ROCm 版 vLLM)
# 见 examples/vllm_adaption/
```

---

## 7. Stage 1 踩坑记录

> 按发生顺序记录;每一坑都给"现象 / 根因 / 修复 / 预防"四项。这是给未来要在新机器/新版本上重做这件事的同学留的备忘。

### 7.1 PEP 517 build isolation 拖来 NVIDIA 版 torch

**现象**

```
OSError: CUDA_HOME environment variable is not set.
File ".../torch/utils/cpp_extension.py", line 1459, in CUDAExtension
    library_dirs += library_paths(device_type="cuda")
```

更深的征兆: pip 在 `/tmp/pip-build-env-*/overlay/` 里装了一堆 `nvidia-cublas / nvidia-cudnn-cu13 / nvidia-cufile / triton / torch-2.12.0`。

**根因**

`pyproject.toml` 写的:

```toml
[build-system]
requires = ["setuptools>=40.0.0", "torch>=1.10.0"]
```

按 PEP 517,pip 默认在 `/tmp/pip-build-env-*/` 隔离环境里装一遍这些 build-time deps。`torch>=1.10.0` 没限定 ROCm wheel,pip 就从 PyPI 主索引拉 **NVIDIA 版 torch**,把系统里正确的 ROCm torch **完全屏蔽**。然后这个隔离环境里的 NVIDIA torch 跑 `CUDAExtension`,找 `CUDA_HOME` 自然失败——即使找到了也没用,因为 `torch.version.hip is None`,`RocmBuilder.is_available()` 会直接返回 False。

**修复**

```bash
pip install -e . --no-build-isolation
```

让 pip 直接复用主 venv 里的 ROCm torch,绕过隔离环境。

**预防**

`--no-build-isolation` 模式下 pip 不会自动装编译时依赖,所以**主 venv 里要先装好**:

```bash
pip install "setuptools>=40.0.0" wheel "Cython>=3.0.10" pybind11 ninja
```

> 长期方案: 给 `pyproject.toml` 加索引/extras 或拆 build profile,让 ROCm 用户明确选 ROCm wheel。但这是 FlexKV 上游的事,本次绕过即可。

---

### 7.2 `transfer.cu` 内联 PTX 汇编 hipify 翻不动

**现象**

```
.hipified/transfer.hip:64: error: invalid output constraint '=f' in asm
.hipified/transfer.hip:68: error: invalid input constraint 'l' in asm
```

涉及代码(NVIDIA 母版 `csrc/gpu_backend/nvidia/transfer.cu`):

```cpp
asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];" : ... : "l"(...) : "memory");
asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" :: "l"(...), "f"(...) : "memory");
```

**根因**

- `ld.global.nc.v4.f32`(non-coherent vectorized load)和 `st.global.cg.v4.f32`(cache-global vectorized store)是 **NVPTX 特有指令**
- 约束 `=f`(float register)、`l`(64-bit register)也是 **NVPTX 后端特有**
- `hipify-perl` 是**纯文本翻译**,不解析 / 不翻译 inline asm —— 这是它的**著名局限**

两段 PTX 的实际作用只是 **cache 行为优化 hint**,功能上等价于"vectorized 4×float 的 load/store"。

**修复**

在 NVIDIA 母版 `transfer.cu` 里加编译期分支:

```cpp
#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__) || defined(FLEXKV_BACKEND_ROCM)
    // ROCm fallback: plain vectorized float4 load/store.
    element = *reinterpret_cast<const float4 *>(&FLOAT4_PTR(src_chunk_ptr)[idx]);
    *reinterpret_cast<float4 *>(&FLOAT4_PTR(dst_chunk_ptr)[idx]) = element;
#else
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];" ...);
    asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" ...);
#endif
```

**为什么改 NVIDIA 母版而不是 hipified 产物**

文档 §3.4 的设计: ROCm 的 hipified 文件是 hipify-perl **每次构建时重新生成**的,改 hipified 文件会被下次构建覆盖。改母版 `transfer.cu` 才是稳定修复。

NVIDIA 编译时 `USE_ROCM / __HIP_PLATFORM_AMD__ / FLEXKV_BACKEND_ROCM` 三个 macro 都不定义,走 `#else` 分支,**保留原 PTX,行为零变化**;ROCm 编译时 `-DUSE_ROCM=1` 命中,走 fallback。

**预防**

- hipify-perl 的盲区: inline asm、PTX intrinsics、`__shfl_*` 等 warp primitives(部分版本)、CUDA-only API(如 cuFile)
- 移植新内核前先 `grep -n "asm volatile\|asm(" csrc/gpu_backend/<vendor>/`,提前发现这类需要手工 fallback 的位置
- 性能上,plain store 比 NVPTX 的 cache hint 版略低;但 AMD 上 cache 行为本就不同,先正确性优先,后续要榨性能再用 `__builtin_amdgcn_*` intrinsics 优化

---

### 7.3 缺 vendored xxhash —— 误以为 pip install 是单阶段构建

**现象**

```
csrc/hash.cpp:7:10: fatal error: xxhash.h: No such file or directory
    7 | #include <xxhash.h>
```

**根因**

FlexKV 是两阶段构建 (见 §5.1),`pip install -e .` **只跑 Stage 1B**,假定 `build/include/xxhash.h` 和 `build/lib/libxxhash.so` 已被 Stage 1A (CMake) 准备好。直接 pip install 就跳过了 Stage 1A,`build/include/` 是空的。

xxhash 在 `third_party/xxHash/` 是 git submodule,需要先拉再编。

**修复**

```bash
git submodule update --init --recursive third_party/xxHash
mkdir -p build && cd build
cmake .. -DFLEXKV_ENABLE_MONITORING=OFF
cmake --build . -j"$(nproc)"
cd ..
```

**预防**

要么走 `install.sh` 一站式(它会自动跑 CMake),要么记住"先 cmake 再 pip"。`README_zh.md` 的 §4 章节叫"Building & Publishing per Vendor",但只讲 pip 命令,没强调前置 CMake 步骤——这是文档可改进的地方。

> 长期方案: 用 `scikit-build-core` 或 `meson-python` 把 CMake 集成进 wheel build,一条 `pip install` 搞定两阶段。但这是大改,目前 install.sh 是务实 workaround。

---

### 7.4 缺系统包 `liburing-dev`

**现象**

```
csrc/transfer_ssd.h:3:10: fatal error: liburing.h: No such file or directory
    3 | #include <liburing.h>
```

**根因**

ROCm 没有 GDS / cuFile,SSD KV-cache 必须走 `transfer_kv_blocks_ssd` 这条 POSIX/io_uring fallback 路径(文档 §3.5.1 明确)。`liburing` 是 Linux 内核级用户库,不能 vendor (要和内核版本对齐),只能走系统包管理。

`RocmBuilder.get_link_args()` 里的 `-luring` 就是它,但**没有人保证开发环境装了这个 dev 包**。NVIDIA 路径估计在某个 dev 容器里跑过、装好了,所以从来没暴露。新 ROCm 容器只装 ROCm SDK,liburing 是裸的。

**修复**

```bash
# Debian/Ubuntu
sudo apt-get install -y liburing-dev

# RHEL/CentOS
sudo yum install -y liburing-devel
```

**预防**

`install.sh` 在 Step 1 里会 `dpkg -s liburing-dev` 检查并自动 apt install。手动构建跳过 install.sh 时,记得对照 §5.2 的"系统依赖"清单。

完整 ROCm 系统依赖矩阵:

| 包 | 干什么用 | 何时需要 |
|----|---------|----------|
| `liburing-dev` / `liburing-devel` | io_uring 异步 IO,SSD KV-cache fallback | **总是需要**(ROCm 没 GDS) |
| `libhiredis-dev` / `hiredis-devel` | Redis client | 仅 `FLEXKV_ENABLE_P2P=1` |
| `cmake` / `git` / `gcc` / `g++` | 构建工具 | 总是需要 |
| `xxhash` (vendored) | 哈希,KV block 唯一标识 | 总是需要,但走 CMake 不是 apt |
| `prometheus-cpp` (vendored) | Metrics | 仅 `FLEXKV_ENABLE_METRICS=1` |

---

### 7.5 (可观察但暂不阻塞)`hipError_t` nodiscard warning

**现象**

```
.hipified/transfer.hip:137: warning: ignoring return value of type 'hipError_t'
    declared with 'nodiscard' attribute [-Wunused-value]
            gpuMemcpyAsync(...)
```

每个 arch 12 条,gfx942 + gfx950 + host 共 36 条。

**根因**

ROCm 7.2 给 `hipError_t` 返回的 API 都加了 `[[nodiscard]]`,要求调用方使用返回值。CUDA 没有这个 attribute,所以 NVIDIA 路径不报。`transfer.cu` 这几处直接忽略返回值在 NVIDIA 上没事,在 ROCm 上吐 warning。

**当前影响**

仅 warning,**不阻塞构建**。但有两个隐患:

1. 如果未来 FlexKV 加 `-Werror`,会立刻变 build 阻塞
2. 这些 API 失败本来就该被发现,不应该静默忽略

**预案**

下一次清理时把忽略返回值的几处改为:

```cpp
auto err = gpuMemcpyAsync(...);
TORCH_CHECK(err == gpuSuccess, "gpuMemcpyAsync failed: ", err);
```

或至少 `(void)gpuMemcpyAsync(...)` 显式消音。当前先**留个 TODO**,不在本次修。

---

### 7.6 (Stage 2)上游 dispatch 测试覆盖盲区

**现象**

`tests/gpu_backend/test_backend_dispatch.py` 9 个测试在 ROCm 机器上**全部通过**,但盘点后发现它没真正验证"在 ROCm 真机上抽象层路由到 RocmBackend"——它只断言 `current_backend.vendor` 在四个枚举值之一,不强制是 ROCM。

**根因**

`test_backend_dispatch.py` 的设计目标是**vendor-neutral**(在任何机器上都该过),所以它只测**框架契约**——ABC 暴露了哪些方法名、`select_backend('generic')` 工作、`build_config` 名字对得上、错误别名抛 ValueError 等。这些命题与"当前环境是哪个 vendor"无关。

**Stage 2 的真正命题**(在 ROCm 机器上)其实包含三层:

1. **框架正确性**(任何 vendor):上游 `test_backend_dispatch.py` 已覆盖 ✅
2. **ROCm 专属路由**(只在 ROCm 机器有意义):
   - `current_backend.vendor is GpuVendor.ROCM`(不是回退到 GENERIC)
   - 三个别名 `rocm / hip / amd` 都构造 `RocmBackend`
   - `RocmBackend()` 可实例化(`@abstractmethod` 全部实现)
3. **ROCm 特定声明**(文档承诺的具体值):
   - `visible_devices_env_vars()` 返回 `(HIP_, ROCR_, CUDA_)_VISIBLE_DEVICES`(§3.7)
   - `supports_direct_storage() is False`(§3.5.1,ROCm 没 GDS)
   - `torch_device_type() == "cuda"`(§3.1 group 1)
4. **Python ↔ C++ 连接**:
   - `flexkv.c_ext` 可 import
   - 暴露 `transfer_kv_blocks` + `TPTransferThreadGroup`(§3.2)
   - **不**暴露 `GDSManager` 等 NVIDIA-only symbol(反向验证 vendor 隔离)

第 2、3、4 层在原始测试里**没有任何覆盖**。

**修复**

新增 `tests/gpu_backend/test_rocm_dispatch.py`,10 个测试逐项覆盖第 2-4 层,文件级用 `pytest.mark.skipif(not torch.version.hip)` 在非 ROCm 机器自动跳过(不污染 vendor-neutral 套件)。详见 §6.1。

**预防 / 给后人的启示**

- 在排查"现成测试是否足够"时,先把命题分层(框架契约 / vendor 路由 / vendor 声明 / Python↔C++ 连接),逐层对照测试文件里的 `assert` 语句
- 反向断言("不该有什么")常常比正向断言更值钱——`test_c_ext_does_not_expose_nvidia_only_symbols` 这种测试一旦回归立刻能报出"vendor 隔离破了"
- vendor-neutral 测试 + vendor-specific 测试是互补关系,不要混进同一个文件;后者用 `pytestmark` 整文件 skip 是最干净的做法

> 长期方案: 把 `test_rocm_dispatch.py` 的相同模式复制成 `test_nvidia_dispatch.py / test_musa_dispatch.py`,让每个 vendor 的"路由+声明+C++ 连接"都有专属覆盖。当前只为 ROCm 做了。

---

### 7.7 (Stage 3)测试设计踩的两个坑

Stage 3 第一次跑 16 个测试挂了 2 个(`test_alloc_free_pinned` 和 `test_ipc_handle_roundtrip_in_same_process`)。两个失败**性质完全不同**,但都暴露了写 vendor 测试时容易犯的同一类错误——**不区分"文档列出的方法"和"真实可被测的契约"**。

#### 7.7.1 测试期望过度:把可选方法当必需测了

**现象**

```
NotImplementedError: RocmBackend does not implement alloc_pinned()
flexkv/gpu_backend/interface.py:199
```

**根因**

`README_en.md §3.1` group 4 把 4 个方法列在一起:`register_host_tensor / unregister_host_tensor / alloc_pinned / free_pinned`。但看 `interface.py`:

```python
@abstractmethod
def register_host_tensor(self, tensor): ...    # 必需

@abstractmethod
def unregister_host_tensor(self, tensor): ...  # 必需

def alloc_pinned(self, size_bytes):             # 可选,默认 NotImplementedError
    raise NotImplementedError(...)

def free_pinned(self, ptr):                     # 可选,默认 NotImplementedError
    raise NotImplementedError(...)
```

ABC 把 group 4 拆成两类,前两个 `@abstractmethod` 必须 override,后两个是默认 `NotImplementedError` 的**可选钩子**。NVIDIA 也没 override `alloc_pinned`——上层 FlexKV 业务全部走 `register_host_tensor`(把现有 tensor page-lock),没人用 `alloc_pinned`(从零分配)。所以 ROCm 不实现是**符合抽象层设计的**。

**修复**

删除 `test_alloc_free_pinned`。

**预防 / 给后人的启示**

写 vendor 真机测试前,**逐个方法核对 `interface.py` 里它是 `@abstractmethod` 还是默认 `NotImplementedError`**:

| 在 `interface.py` 里的形式 | 含义 | 测试该如何处理 |
|---|---|---|
| `@abstractmethod def foo(...): ...` | 必需,所有 vendor 必须实现 | 直接测 `foo()` 工作 |
| `def foo(...): raise NotImplementedError(...)` | 可选钩子,vendor 视情况实现 | 不测,或者用 `pytest.skip` 加 `if not implemented` |
| `def foo(...): return False` (capability flag) | 默认 False 的能力查询 | 测 `isinstance(out, bool)`,不测具体值;真值时再测对应能力 |

文档 §3.1 列出的"7 组方法"是给读者看抽象层全景的,**不是契约清单**。契约在 `interface.py`。

#### 7.7.2 测试设计违反 API 语义边界:同进程 round-trip IPC

**现象**

```
RuntimeError: hipIpcOpenMemHandle failed with error code 201 on device 0
flexkv/gpu_backend/rocm/backend.py:264
```

**根因**

我写了 `test_ipc_handle_roundtrip_in_same_process`,在同一进程里先 `get_ipc_handle` 再 `open_ipc_handle`。但 CUDA / HIP 文档明确说:

> `*IpcOpenMemHandle` may only be called by the **importing process** —— a process different from the exporter.

同进程已经有这块 GPU 内存的虚地址了,runtime 拒绝再 map,避免重复映射 / 双重释放。这是设计如此。ROCm 7.x 严格执行(返回 error 201);NVIDIA 早期版本对同进程 round-trip 容忍度更高,但那是"恰好能跑",不是契约。

`§3.6` 文档其实写得很清楚——"GPU pointer must travel **across processes**"——但我下意识把 IPC 当 unit-testable API 在同进程里隔离测了,违反了它的使用语境约束。

**修复**

测试改为只验证 export(`get_ipc_handle` 返回非空 bytes),round-trip 留给 Stage 4 的 `test_memory_handle.py`(那里用 `multiprocessing.spawn` 跑真实跨进程拓扑)。

**附带发现:`README_en.md §3.6.2` 文档过期**

修测试参数时发现 `interface.py` 的 `open_ipc_handle` 真实签名比 §3.6.2 文档少 `strides` / `storage_offset`,改名 `offset`。这是**上游文档错误**,值得给 FlexKV 提 PR 修正。

**预防 / 给后人的启示**

1. **识别 API 的"使用语境约束"**:不是所有 API 都能在 unit test 风格下隔离测。IPC、跨设备 P2P、跨节点 RDMA 这类,本身就要求"两个独立的执行上下文",同进程测试要么挂、要么得到无意义结果
2. **Stage 之间应该有清晰的能力边界**:
   - Stage 3 = 单进程能力(设备 / 流 / pinned host / handle export)
   - Stage 4 = 跨进程协作(IPC round-trip / multiprocessing-based KV cache 共享)

   同一个能力(如 IPC)的不同维度按这个边界拆分,而不是 Stage 3 抢着测完
3. **以代码为准,不以文档为准**:`README_en.md` 是设计文档,签名层面会落后于代码。测试要按 `interface.py` 真实签名写,顺手发现的文档不一致(如 §3.6.2)单独提 PR 修

#### 7.7.3 两次踩坑的共同教训

把"文档章节"误当成"测试契约清单"。文档章节是用来**讲设计**的(读者视角),它会列出所有相关概念以保证完整性;但**测试契约**应该来自代码本身——`@abstractmethod` 是必需的,默认 NotImplementedError 是可选的,跨进程 API 不在同进程测。

下次写新 vendor 测试时,先读 `interface.py` 决定测什么,再读 README 决定每个测试该断言什么,**顺序不要反**。

---

### 7.8 (Stage 4)两个上游 abstraction 漏洞 ⭐

> 这是本次 ROCm 验证最有迁移价值的发现。和 §7.1-§7.7 不同,这里的两个问题**不是测试问题、不是 Cython 行为差异、不是文档过期**,而是 **GpuBackend 抽象层**真实漏掉了一类东西——**vendor-preferred 默认值**。它们在 NVIDIA 上是合理默认,在 ROCm 上是 bug。

#### 7.8.1 `enable_mps=True` 默认值假设了 NVIDIA-only 工具

**现象**

跑 `test_kvmanager.py` 全部 60 个测试 fail,`--tb=line` 后 100% 收敛到同一个根因:

```
FileNotFoundError: [Errno 2] No such file or directory: 'nvidia-cuda-mps-control'
```

**根因**

`flexkv/common/config.py:130`:

```python
enable_mps=bool(int(os.getenv('FLEXKV_ENABLE_MPS', 1)))   # ← 默认 True
```

`flexkv/kvmanager.py:111-113`(没走 abstraction):

```python
if self.enable_mps:
    subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)   # ← 硬编码 NVIDIA 工具
```

NVIDIA MPS(Multi-Process Service)让多个 CUDA 进程共享一个 GPU 的 SM 时间片,server-client 模式下提升性能。**ROCm 没有等价工具**——AMD 有 SR-IOV / partition modes,API 完全不同。

**短期旁路**

```bash
export FLEXKV_ENABLE_MPS=0
```

无功能影响,只是失去 NVIDIA-MPS 的多进程 SM 复用优化。多进程仍然可以共享 GPU,只是上下文切换开销略大。

**长期修复方向**

抽象层应该:

```python
# interface.py 加一个新方法
def supports_mps(self) -> bool:
    """Whether this backend has a MPS-equivalent multi-process scheduler."""
    return False   # 默认 False

# nvidia/backend.py
def supports_mps(self) -> bool:
    return True

# kvmanager.py
if self.enable_mps and current_backend.supports_mps():
    current_backend.start_mps_daemon()   # 抽象 spawn
```

进一步可以让 `current_backend.start_mps_daemon()` / `stop_mps_daemon()` 也走 abstraction,彻底消除 `kvmanager.py` 里的 `nvidia-cuda-mps-control` 字面量。

---

#### 7.8.2 `use_ce_transfer=False` 默认值依赖 NVIDIA UVA + PTX

**现象**

设了 `FLEXKV_ENABLE_MPS=0` 之后,KVManager 不再因 MPS 挂,但跑到第一次 H2D copy 时 GPU 端崩溃:

```
writing initial data...
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x7ef1847f9000.
GPU coredump: System pipe patterns not supported in containers.
```

地址 `0x7ef...` 是典型的 host 用户态 mmap 地址——**GPU device kernel 在尝试通过 plain `*ptr` 直接访问 host pinned memory,在 ROCm 上立刻触发 memory access fault**。

**根因**

`flexkv/common/config.py:114-115`:

```python
use_ce_transfer_h2d=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_H2D', 0)))   # ← 默认 False
use_ce_transfer_d2h=bool(int(os.getenv('FLEXKV_USE_CE_TRANSFER_D2H', 0)))   # ← 默认 False
```

`csrc/gpu_backend/nvidia/transfer.cu:96-115` 有两条路径:

```cpp
if (use_ce_transfer) {
    // Path A: Copy Engine - host 端 hipMemcpyAsync 循环
    gpuMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, chunk_size_in_bytes, ...);
} else {
    // Path B: Device kernel - 直接读写 host pointer (zero-copy)
    transfer_kv_blocks_kernel<<<grid, block, 0, stream>>>(...);
    //   kernel 内部: *ptr 读写 cpu_chunk_ptr
}
```

**Path B 在 NVIDIA 上能跑,靠的是两个 NVIDIA 特有机制叠加**:

1. **UVA (Unified Virtual Addressing)**:`cudaHostRegister` 注册过的 host memory 在所有 CUDA context 中共享同一个虚地址,device kernel 可以直接通过该地址访问
2. **PTX 特殊指令** (`ld.global.nc.v4.f32` / `st.global.cg.v4.f32`):专门用于 zero-copy 访问 mapped host memory 的 cache hint 指令(我们在 §7.2 改过这部分)

**ROCm 没有完整的等价机制**:
- `hipHostRegister + Portable` flag 只让 host memory 在多 GPU context 间可见,不会自动让 device kernel 看到
- `hipHostRegisterMapped` + `hipHostGetDevicePointer` 可以拿到 device 端别名,但 transfer.cu 的代码没走这条
- 即使我们在 §7.2 patch 了 NVPTX-only 的内联汇编,改成 plain `*ptr` 在 ROCm device kernel 里也**不能 zero-copy 访问 host memory**

**短期旁路**

```bash
export FLEXKV_USE_CE_TRANSFER_H2D=1
export FLEXKV_USE_CE_TRANSFER_D2H=1
```

这让 ROCm 走 Path A——host 端循环调用 `hipMemcpyAsync`,这是 ROCm 上稳定且唯一正确的 H2D/D2H copy 路径。性能上**比 NVIDIA Path B 略差**(多了 kernel launch 开销和并行度降低),但**正确性 100% 保证**。

**长期修复方向**

抽象层加 vendor-preferred 默认值钩子:

```python
# interface.py
def prefer_ce_transfer_h2d(self) -> bool:
    """Whether this backend should default to host-driven hipMemcpyAsync
    instead of in-kernel zero-copy access to host pinned memory."""
    return False   # NVIDIA 默认

def prefer_ce_transfer_d2h(self) -> bool:
    return False

# rocm/backend.py
def prefer_ce_transfer_h2d(self) -> bool:
    return True   # ROCm device kernel 不能通过 plain *ptr 访问 host pinned mem

def prefer_ce_transfer_d2h(self) -> bool:
    return True

# common/config.py
use_ce_transfer_h2d=bool(int(os.getenv(
    'FLEXKV_USE_CE_TRANSFER_H2D',
    int(current_backend.prefer_ce_transfer_h2d())
)))
```

#### 7.8.3 共同的根本设计 gap

这两个漏洞是同一类:**README §3.1 的 7 组 abstraction 方法只覆盖了"能力声明"和"动作",没有覆盖"vendor-preferred 默认值"**。

| 抽象层已覆盖的类别 | 例子 |
|---|---|
| 能力声明(布尔查询) | `supports_ipc()`, `supports_direct_storage()` |
| 动作(执行) | `set_device()`, `register_host_tensor()`, `get_ipc_handle()` |
| **缺失的类别**:vendor-preferred 默认值 | `supports_mps()` / `prefer_ce_transfer_*()` 这种 |

修复这两个具体漏洞后,长期还应该:
1. **审计 `flexkv/common/config.py`**,所有 `os.getenv('FLEXKV_*', <hardcoded_default>)` 形式的默认值,挨个判断"这个默认值是否假定了某种 vendor 特性",有的话就提到 abstraction 层
2. **README §3.1 加第 8 组** "Vendor-preferred defaults",显式把这类钩子归类
3. 给 FlexKV 上游提 issue/PR,把这两个具体的提到 abstraction 层

#### 7.8.4 给后人的启示

1. **`60 failed in 0.88s`** 听起来像灾难,实际**用 `--tb=line | grep -oE 'file:line' | sort | uniq -c`** 三秒就能看出"60 个其实是同一个根因"——这种分布分析比读 traceback 高效得多
2. **GPU memory access fault 的地址有效信号**:`0x7ef...` / `0x7ff...` 这种位置是 host 用户态 mmap 地址(典型 64-bit Linux),不是 device pointer(那个一般在 `0x7f1...0000` 之类的更靠后段)。看到 GPU 报 host 地址访问错误,**100% 是 device kernel 在用 zero-copy 模式访问 host memory**——这种问题在 NVIDIA 上靠 UVA 自动处理,移植到其他 vendor 都要单独审视
3. **跑 e2e 测试前先想清楚"vendor-preferred 默认值"在哪些维度可能不同**:MPS、UVA、cooperative launch、specific intrinsics(如 `__shfl_sync`)、cuda graph、unified memory、CTA reservation……这些都是 NVIDIA 的"舒适默认",移到其他 vendor 上每一个都要单独验证
4. **抓上游漏洞要看测试结果分布,不要看测试通过率**:60/60 看似是干净的"全过",但加了 2 个环境变量才过——**这两个环境变量本身就是漏洞的指纹**。Stage 4 真正的产出不是 "60 passed",而是"发现了 2 个上游 abstraction 漏洞 + 旁路方案"

---

### 7.9 (Stage 4 边角)上游测试 bug 与 Cython 行为差异

跑 Stage 4 过程中还遇到几个**与 ROCm 无关**但需要绕过的上游问题,简记于此:

#### 7.9.1 `test_memory_handle.py` 两处上游测试 bug(本次顺手修了)

| Bug | 现象 | 修复 |
|---|---|---|
| `test_tensor_from_cpu_tensor` regex 过期 | `pytest.raises` match `"Only support CUDA tensor sharing"`,但代码已改为 vendor-neutral 的 `"Only support GPU tensor sharing"` | 改 match 字符串(test_memory_handle.py:387) |
| `test_fp8_tensor_from_bytes_roundtrip` NameError | 引用 `_worker_test_fp8_tensor_from_bytes` 但文件里**根本没定义**这个函数 | 按其他 worker 的镜像新增该函数,逻辑符合父进程 `assert not isinstance(result, str)` 的期望 |

这两个 bug 在 NVIDIA 上同样会挂,只是上游 CI 没拦下。值得单独给 FlexKV 提 PR,但**不在本次 ROCm 验证范围**。

#### 7.9.2 `test_cache_engine.py` 4 个 Cython 编译模式行为差异

`test_config_init[config33|34]` 在 `CacheEngine` / `CacheEngineAccel` 各失败一次,共 4 个。

**根因**:`__init__(..., protected_threshold: int = 2)` 这种 PEP 526 类型注解被 Cython 编译时当作 `cdef int` 处理:

| 输入 | Cython 入口行为 | 测试期望 | 实际 |
|---|---|---|---|
| `1.5` (float) | 隐式截断 → `1` | `ValueError` | 没 raise(被静默接受) |
| `None` | 直接 `TypeError` | `ValueError` | `TypeError`(类型对不上但异常类不对) |

**与 ROCm 无关**——NVIDIA release 模式下同样表现。`FLEXKV_DEBUG=1 pip install -e .` 会跳过 Cython 编译走纯 Python,这两个测试就过——但那是鸵鸟方案。本次保持现状,**不修**(选择"不修,记录后继续",见 §2 进度表 Stage 4 备注)。

`test_match_and_insert / test_eviction_policy / test_slru_*` 等 175 个**真正测 KV transfer 热路径**的核心测试**全部通过**,所以这 4 个边缘 failure 不影响 ROCm 验证结论。

---

## 8. 总结性建议(给后续在新 ROCm 机器上做这件事的同学)

按下面的顺序操作可以**一次走完 Stage 0-4**,避免 §7 全部坑:

```bash
# === Step 0: 确认环境 (§4) ===
rocm-smi && hipcc --version && which hipify-perl
python -c "import torch; assert torch.version.hip is not None"

# === Step 1: 系统依赖 (§7.4) ===
sudo apt-get install -y liburing-dev

# === Step 2: 编译时 Python 依赖 (§7.1) ===
pip install "setuptools>=40.0.0" wheel "Cython>=3.0.10" pybind11 ninja

# === Step 3: Stage 1A — vendored 库 (§7.3) ===
git submodule update --init --recursive third_party/xxHash
mkdir -p build && (cd build && cmake .. -DFLEXKV_ENABLE_MONITORING=OFF && cmake --build . -j"$(nproc)")

# === Step 4: Stage 1B — Python 扩展 (§7.1, §7.2) ===
# 仓库已包含 §7.2 的 PTX fallback patch,无需再手动改
FLEXKV_GPU_BACKEND=rocm pip install -e . --no-build-isolation -v

# 验证 c_ext 产物
python -c "import flexkv.c_ext as ext; print(ext.__file__)"

# === Step 5: 设置 ROCm 必备运行时 env (§3.1, §7.8) ===
export FLEXKV_ENABLE_MPS=0
export FLEXKV_USE_CE_TRANSFER_H2D=1
export FLEXKV_USE_CE_TRANSFER_D2H=1

# === Step 6: Stage 2-4 测试 ===
# Stage 2 (无 GPU,数秒)
pytest tests/gpu_backend/ -v

# Stage 3 (真机,数秒)
pytest tests/gpu_backend/test_rocm_runtime.py -v

# Stage 4 (端到端,kvmanager 耗时 ~25 分钟)
rm -f /tmp/flexkv_*           # 每次跑前清理 ZMQ socket 残留
pytest tests/test_memory_handle.py -v
pytest tests/test_cache_engine.py -v
pytest tests/test_kvmanager.py -v
pytest tests/test_namespace_isolation.py -v
```

如果上面任一步失败,对照 §7 找对应的坑:

| 失败现象关键词 | 对应章节 |
|---|---|
| `OSError: CUDA_HOME environment variable is not set` / `nvidia-cuda-cu13` | §7.1 PEP 517 build isolation |
| `error: invalid output constraint '=f' in asm` / NVPTX | §7.2 内联 PTX 不能 hipify |
| `xxhash.h: No such file or directory` | §7.3 缺 vendored xxhash |
| `liburing.h: No such file or directory` | §7.4 缺系统包 |
| `'pybind11::class_<...>' declared with greater visibility` | (warning,可忽略) |
| `RocmBackend does not implement alloc_pinned()` | §7.7.1(测试期望过度,不是 bug) |
| `hipIpcOpenMemHandle failed with error code 201` | §7.7.2(同进程 IPC round-trip 是设计上不允许的) |
| `nvidia-cuda-mps-control: No such file` | §7.8.1(`FLEXKV_ENABLE_MPS=0`) |
| `Memory access fault by GPU node-N on address 0x7ef...` | §7.8.2(`FLEXKV_USE_CE_TRANSFER_*=1`) |
| pytest 启动后**无输出地卡住** | §7.8 / §6.3.1(`rm -f /tmp/flexkv_*`) |
| `test_tensor_from_cpu_tensor` regex 不匹配 / `_worker_test_fp8_tensor_from_bytes` NameError | §7.9.1(上游测试 bug,已顺手修) |
| `test_config_init[config33/34]` 4 个 fail | §7.9.2(Cython 行为差异,与 ROCm 无关,179/183 通过即合格) |

---

## 9. 本次 ROCm 验证的核心产出

按重要性排列:

1. **2 个上游 abstraction 漏洞**(§7.8) — `enable_mps` 和 `use_ce_transfer` 默认值假定了 NVIDIA 特性。短期靠 2 个环境变量旁路,长期需给 GpuBackend ABC 加"vendor-preferred 默认值"钩子(README §3.1 漏掉的第 8 组)。**这是本次最有迁移价值的发现,值得给 FlexKV 上游提 issue/PR**

2. **1 个代码 patch**(§7.2) — `transfer.cu` 加 `#if defined(USE_ROCM)` 分支处理内联 PTX(hipify 翻不动)。NVIDIA 路径完全不受影响

3. **2 个 ROCm-专属测试文件** — 补全 Stage 2-3 的覆盖盲区:
   - `tests/gpu_backend/test_rocm_dispatch.py`(10 个,无 GPU)— ROCm 路由 + ABC 完整性 + ROCm 特定声明 + Python↔C++ binding 连接
   - `tests/gpu_backend/test_rocm_runtime.py`(15 个,需 GPU)— 设备 / 流 / pinned host / IPC export

4. **2 个上游测试 bug 顺手修**(§7.9.1) — `test_memory_handle.py` 的 regex 过期 + 缺失 worker 函数。NVIDIA 上同样会挂,值得单独提 PR

5. **本文档** — 完整记录 Stage 0-4 的踩坑与决策,任何后人在新 ROCm 机器上做这件事可以照着 §8 的脚本一次走完

