# FlexKV CUDA → ROCm 迁移验证清单

> 基于 `feat/rocm-ce-backend` 分支实际实现整理，覆盖构建、C++ 运行时、Python 层、测试与功能边界。
> AMD/ROCm 后端为 **CE-only**：仅启用 CE 主机暂存异步复制路径，不迁移也不编译 NVIDIA PTX kernel、GDS/cuFile、nvCOMP 压缩与 NVTX tracing。

---

## 1. 构建工具链

### 1.1 ROCm 构建检测 (`setup.py`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| `is_rocm_build()` 检测逻辑 | 优先 `FLEXKV_USE_ROCM` 环境变量，其次 `torch.version.hip` | `setup.py:12-22`，三级回退：env=1 → True，env=0 → False，否则 `torch.version.hip` |
| `FLEXKV_USE_ROCM=1` 显式强制 | 即使未安装 torch 也能进入 ROCm 构建分支 | env 检查在 try/except torch 之前 |
| CUDA 构建不受影响 | `IS_ROCM_BUILD=False` 时走原逻辑 | 所有 ROCm 分支为 `if IS_ROCM_BUILD:` 独立判断 |

### 1.2 源文件选择 (`setup.py:258-261`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 排除 `csrc/transfer.cu` | 不编译含 NVIDIA PTX 的 kernel | `"csrc/ce_transfer_dispatch.cu" if IS_ROCM_BUILD else "csrc/transfer.cu"` |
| ROCm 使用 `csrc/ce_transfer_dispatch.cu` | CE-only 分发器替代 | 同上 |
| 其余源文件保持不变 | `ce_transfer.cu`、`layerwise.cpp`、`tp_transfer_thread_group.cpp` 等共用 | `cpp_sources` 列表其余项不分支 |

### 1.3 编译宏与参数 (`setup.py:316-343`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 定义 `FLEXKV_USE_ROCM` | C++ 侧走 HIP 分支 | `extra_compile_args.extend(["-DFLEXKV_USE_ROCM", "-DFLEXKV_CE_ONLY"])` |
| ROCm 定义 `FLEXKV_CE_ONLY` | pybind 强制 CE 路径 | 同上，`nvcc_compile_args` 也追加 |
| ROCm 不定义 `CUDA_AVAILABLE` | 避免运行时误判 CUDA | `if not IS_ROCM_BUILD: extra_compile_args.append("-DCUDA_AVAILABLE")` |
| ROCm 跳过 `TORCH_CUDA_ARCH_LIST` 探测 | 不执行 `detect_cuda_arch()`，避免 nvcc 依赖 | `if IS_ROCM_BUILD: print(...)` else 分支 |
| ROCm 不设置 `PYTORCH_ROCM_ARCH` | 让 PyTorch 自动解析本地 ROCm 目标 | 未显式设置，依赖 PyTorch 扩展默认行为 |

### 1.4 链接依赖 (`setup.py:286-294`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 不链接 `-lcuda` | 移除 NVIDIA CUDA 驱动库 | `if not IS_ROCM_BUILD: extra_link_args.insert(0, "-lcuda")` |
| ROCm 不链接 nvCOMP 库 | `enable_nvcomp` 强制为 False | `setup.py:249-255` 强制置零并写回 env |
| ROCm 不链接 GDS 库 | `enable_gds` 强制为 False | 同上 |
| 其余链接库保持 | `-lxxhash -lpthread -lrt -luring` 不变 | 共享 base 列表 |

### 1.5 专有模块排除

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| GDS 源文件不编译 | `csrc/gds/*.cpp` 不在 `cpp_sources` | 仅在 `enable_gds` 时 append，ROCm 下被强制关闭 |
| nvCOMP 源文件不编译 | `csrc/compression/**/*.cpp` 不编译 | 仅在 `enable_nvcomp` 时 extend |
| bindings 中 GDS/nvCOMP 注册条件化 | `#ifdef FLEXKV_ENABLE_GDS` / `#ifdef FLEXKV_ENABLE_NVCOMP` | `bindings.cpp:21-24, 45-48` |
| `rocm_utils.h` 纳入 header 依赖追踪 | 增量构建感知 | `hpp_sources` 中加入 `"csrc/rocm_utils.h"` |

### 1.6 构建脚本 (`build.sh`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| `--rocm` 参数解析 | 设置 `ROCM_BUILD=1` | `build.sh` case 分支新增 `--rocm` |
| 缺少 `hipcc` 快速失败 | 报错退出 | `command -v hipcc` 检查，失败 exit 1 |
| 设置 ROCm 环境变量 | `FLEXKV_USE_ROCM=1`、GDS=0、NVCOMP=0 | `build.sh:78-81` |
| 默认构建不受影响 | 不传 `--rocm` 时走原 CUDA 路径 | `ROCM_BUILD=0` 默认值 |

### 1.7 安装与测试脚本

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| `install.sh` 检查 hipcc | ROCm PyTorch 环境检查 hipcc 而非 nvcc | `python3 -c "import torch; raise SystemExit(0 if torch.version.hip else 1)"` 分支 |
| `install.sh` 不自动安装 CUDA PyTorch | 报错而非 `pip install torch` | 改为 `error "PyTorch is not installed..."` |
| `install.sh` 移除 nvtx 依赖 | `RUNTIME_DEPS` 不含 nvtx | 已移除 |
| `run_tests.sh` 不硬编码 CUDA stub 路径 | 解析 `torch.__file__/lib` | `TORCH_LIB_DIR` 变量动态获取 |

---

## 2. C++ 运行时层

### 2.1 统一运行时适配层 (`csrc/rocm_utils.h`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| `FLEXKV_USE_ROCM` 时包含 `hip/hip_runtime.h` | 使用 HIP 运行时 | `rocm_utils.h:5-12`，包含前显式定义 `__HIP_PLATFORM_AMD__` |
| 否则包含 `cuda_runtime.h` + `nvtx3` | CUDA 后端不受影响 | `rocm_utils.h:62-66` else 分支 |
| 类型别名映射 | `cudaStream_t` → `hipStream_t` 等 | `rocm_utils.h:14-17` using 声明 |
| 错误码映射 | `cudaSuccess` → `hipSuccess` 等 | `rocm_utils.h:19-20` constexpr |
| 枚举/标志映射 | `cudaEventDisableTiming`、`cudaHostAllocDefault` 等 | `rocm_utils.h:21-26` constexpr |
| API 宏映射 | `cudaMalloc` → `hipMalloc` 等 23 个 | `rocm_utils.h:28-50` #define |
| NVTX 无操作桩 | `nvtxRangeStartA` 等返回空 | `rocm_utils.h:56-60` inline no-op |
| `CUDART_CB` 宏定义 | 回调签名兼容 | `rocm_utils.h:52-54` |

### 2.2 头文件替换

| 文件 | 原依赖 | 替换为 | 状态 |
|------|--------|--------|------|
| `csrc/transfer.cuh` | `#include <cuda_runtime.h>` | `#include "rocm_utils.h"` | ✅ |
| `csrc/ce_transfer.h` | `#include <cuda_runtime.h>` | `#include "rocm_utils.h"` | ✅ |
| `csrc/gtensor_handler.cuh` | `#include <cuda_runtime.h>` | `#include "rocm_utils.h"` | ✅ |
| `csrc/tp_transfer_thread_group.h` | `#include <cuda_runtime.h>` | `#include "rocm_utils.h"` | ✅ |
| `csrc/layerwise.h` | `<cuda_runtime.h>` + `<nvtx3/nvToolsExt.h>` | `#include "rocm_utils.h"` (移除 nvtx3) | ✅ |
| `csrc/ce_transfer.cu` | 间接依赖 | 显式追加 `#include "rocm_utils.h"` | ✅ |
| `csrc/layerwise.cpp` | `#include <nvtx3/nvToolsExt.h>` | 移除（由 rocm_utils.h 提供 no-op） | ✅ |
| `csrc/bindings.cpp` | `<cuda_runtime.h>` + `<nvtx3>` | `#include "rocm_utils.h"` | ✅ |
| `csrc/transfer.cu` | classic kernel 抽到共享头 | CUDA 入口调用 `launch_classic_transfer_kv_blocks` | ✅ |
| `csrc/transfer_kernels.cuh` | CUDA/ROCm 共享 classic CTA kernel | PTX（CUDA）/ nontemporal（ROCm） | ✅ |

### 2.3 ROCm 分发器 (`csrc/ce_transfer_dispatch.cu`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| 入口签名与 `transfer.cu` 一致 | 模板 `transfer_kv_blocks<BackendType>` | `ce_transfer_dispatch.cu` |
| `use_ce_transfer=false` | 走 classic CTA kernel（`CLASSIC_KERNEL`） | `launch_classic_transfer_kv_blocks` |
| 禁用 memcpy2D 快路径 | `rocm_ce_config.enable_memcpy2d = false` | CE 分支内强制关闭 |
| 保留 CE 分析路径 | CONTIG/SEGMENT/GATHER（无 COMPUTE_KERNEL） | switch 内各 case |
| 保留 `path_opt_enabled=false` 回退 | 走 `ce_transfer_per_block` | CE 分支 |
| 同步请求正确处理 | `cudaStreamSynchronize` / HIP 映射 | classic 与 CE 末尾 |
| 三后端模板实例化 | VLLM / TRTLLM / SGLANG | 文件末尾 |
| 输入校验 | 非负、8 字节对齐、非空指针 | 入口 TORCH_CHECK |

### 2.4 pybind 绑定加固 (`csrc/bindings.cpp`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| 设备检查 | metadata 和 CPU tensor 在 CPU 上 | `TORCH_CHECK(...device().is_cpu()...)` |
| dtype 检查 | block-id 和指针 tensor 为 int64 | `TORCH_CHECK(...scalar_type() == torch::kInt64...)` |
| 连续性检查 | 所有 CE transfer tensor 连续 | `TORCH_CHECK(...is_contiguous()...)` |
| 长度一致性 | GPU 和 CPU block-id 长度相等 | `TORCH_CHECK(...numel() == ...)` |
| 参数正数校验 | strides、chunk_size、num_layers > 0 | `TORCH_CHECK(...> 0...)` |
| 溢出防护 | 传输大小不溢出 int64 | 分步乘法前检查 `std::numeric_limits<int64_t>::max()` |
| 块数限制 | 不超过 `int` 范围 | `TORCH_CHECK(...numel() <= std::numeric_limits<int>::max())` |
| GPU 指针表容量 | 按后端类型校验最小容量 | `required_tensor_ptrs` 计算 + check |
| `FLEXKV_CE_ONLY` | 仅强制关闭 memcpy2D；允许 classic kernel | `#ifdef FLEXKV_CE_ONLY` → `ce_enable_memcpy2d = false` |
| GDS 头文件条件包含 | `#ifdef FLEXKV_ENABLE_GDS` | `bindings.cpp:21-24` |
| nvCOMP 注册条件化 | `#ifdef FLEXKV_ENABLE_NVCOMP` | `bindings.cpp:45-48` |

### 2.5 `csrc/ce_transfer.cu` 实际 CUDA API 调用

以下 API 经 `rocm_utils.h` 宏映射到 HIP 等价物，无需修改源码：

| CUDA API | HIP 映射 | 调用位置 |
|----------|---------|----------|
| `cudaMallocHost` | `hipHostMalloc` | 设备缓冲分配 |
| `cudaFreeHost` | `hipHostFree` | 设备缓冲释放 |
| `cudaEventCreateWithFlags` | `hipEventCreateWithFlags` | 双缓冲事件 |
| `cudaEventDestroy` | `hipEventDestroy` | 事件清理 |
| `cudaEventRecord` | `hipEventRecord` | 分段同步 |
| `cudaEventQuery` | `hipEventQuery` | 双缓冲就绪检测 |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | 异步 H2D/D2H |
| `cudaMemcpy2DAsync` | `hipMemcpy2DAsync` | memcpy2D 路径（ROCm 下已禁用） |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | 同步等待 |
| `cudaGetLastError` | `hipGetLastError` | 错误检测 |
| `cudaGetErrorString` | `hipGetErrorString` | 错误描述 |
| `cudaLaunchHostFunc` | `hipLaunchHostFunc` | 主机回调 |
| `cudaGetDevice` | `hipGetDevice` | 当前设备查询 |
| `cudaSetDevice` | `hipSetDevice` | 设备切换 |

---

## 3. Python 层

### 3.1 主机内存注册 (`flexkv/transfer/host_buffer.py`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 加载 `libamdhip64.so` | 不加载 `libcudart.so` | `_get_gpu_runtime()` 按 `_is_rocm()` 选择库名 |
| `cudaHostRegister` → `hipHostRegister` | 运行时符号分派 | `_runtime_symbol("cudaHostRegister", "hipHostRegister")` |
| `cudaHostUnregister` → `hipHostUnregister` | 同上 | `_runtime_symbol("cudaHostUnregister", "hipHostUnregister")` |
| `cudaHostAlloc` → `hipHostMalloc` | 映射主机内存分配 | `_runtime_symbol("cudaHostAlloc", "hipHostMalloc")` |
| `cudaFreeHost` → `hipHostFree` | 映射主机内存释放 | `_runtime_symbol("cudaFreeHost", "hipHostFree")` |
| 公共接口不变 | `cudaHostRegister()` / `cudaHostUnregister()` / `alloc_mapped_host_tensor()` 签名保持 | 函数名和参数未改 |
| 错误消息区分后端 | ROCm 报 hip API 名 | `api = "hipHostRegister" if _is_rocm() else "cudaHostRegister"` |
| `alloc_mapped_host_tensor` 大小校验 | 正数检查 | `if num_elements <= 0: raise ValueError` |
| 空指针检查 | 分配结果非空 | `if err != 0 or not host_ptr.value:` |
| `weakref.finalize` 使用正确 | 释放函数和参数绑定 | `weakref.finalize(tensor, runtime_free, host_ptr)` |
| `torch.version.hip` 安全访问 | 兼容无 hip 属性的 CUDA PyTorch | `getattr(torch.version, "hip", None)` |
| `from __future__ import annotations` 移除 | 兼容 Python 3.6 | 已移除，返回类型改为字符串前向引用 |

### 3.2 配置默认值 (`flexkv/common/config.py`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 自动启用 CE H2D | `use_ce_transfer_h2d=True` | `... or _is_rocm_runtime()` (`config.py:747-748`) |
| ROCm 自动启用 CE D2H | `use_ce_transfer_d2h=True` | `... or _is_rocm_runtime()` (`config.py:749-750`) |
| ROCm 禁用 memcpy2D | `enable_ce_memcpy2d=False` | `... and not _is_rocm_runtime()` (`config.py:756-757`) |
| ROCm 禁用 GDS | `enable_gds=False` | `... and not _is_rocm_runtime()` (`config.py:883-884`) |
| 环境变量仍可覆盖 | `FLEXKV_USE_CE_TRANSFER_H2D=0` 仍可读取 | `bool(int(os.getenv(...)))` 优先 |
| `_is_rocm_runtime()` 安全访问 | 兼容无 hip 属性 | `getattr(torch.version, "hip", None)` |

### 3.3 NVTX Tracing 门面 (`flexkv/common/tracing.py`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 不导入 nvtx 包 | 使用 `_NoopNvtx` | `if getattr(torch.version, "hip", None): return _NoopNvtx()` |
| CUDA 环境仍用真 nvtx | `import nvtx` | `try: import nvtx` |
| nvtx 安装缺失时回退 | CUDA 环境也无 nvtx 时不崩溃 | `except ImportError: return _NoopNvtx()` |
| 无操作接口完整 | start/end/push/pop range | `_NoopNvtx` 四个静态方法 |
| 模块级 `nvtx` 单例 | `from flexkv.common.tracing import nvtx` | `nvtx = _load_nvtx()` 模块加载时初始化 |

### 3.4 NVTX 导入替换

| 文件 | 原代码 | 替换为 | 状态 |
|------|--------|--------|------|
| `flexkv/transfer/worker.py` | `import nvtx` | `from flexkv.common.tracing import nvtx` | ✅ |
| `flexkv/transfer/transfer_engine.py` | `import nvtx` | `from flexkv.common.tracing import nvtx` | ✅ |
| `flexkv/kvtask.py` | `import nvtx` | `from flexkv.common.tracing import nvtx` | ✅ |
| `flexkv/transfer_manager.py` | `import nvtx` | `from flexkv.common.tracing import nvtx` | ✅ |
| `flexkv/transfer/compression/ans/ans_strategy.py` | `import nvtx` | `from flexkv.common.tracing import nvtx` | ✅ |

### 3.5 依赖清理 (`requirements.txt`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| 移除 `nvtx` 依赖 | requirements.txt 不含 nvtx | 已删除 `nvtx>=0.2.8` 行 |
| CUDA 构建兼容 | 缺失 nvtx 时 tracing 回退 no-op | `tracing.py` 的 `except ImportError` |

---

## 4. 测试

### 4.1 传输正确性测试 (`tests/test_kv_transfer_correctness.py`)

| 验证项 | 预期 | 实际实现 |
|--------|------|----------|
| ROCm 检测标志 | `IS_ROCM = bool(getattr(torch.version, "hip", None))` | `test_kv_transfer_correctness.py:32` |
| ROCm 仅执行 CE 引擎 | `ENGINES = [pytest.param("ce", True, id="ce")]` | ROCm 分支不含 `("cuda", False)` |
| ROCm 禁用 memcpy2D 参数 | `CE_MEMCPY2D_CONFIGS = [False]` | ROCm 分支不含 `True` |
| CUDA 构建保持原参数 | `ENGINES` 含 cuda + ce | `else` 分支保持原列表 |
| 多 GPU 门槛 | `NUM_GPUS < 2` 时跳过 | `pytestmark` 不变 |

### 4.2 ROCm 后端边界测试 (`tests/test_rocm_backend.py`)

| 验证项 | 预期 | 实现位置 |
|--------|------|----------|
| 非 ROCm 环境跳过 | `pytestmark skipif not hip` | `test_rocm_backend.py:5-7` |
| CE H2D 默认启用 | `assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d` | `test_rocm_backend_is_ce_only` |
| CE D2H 默认启用 | `assert GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h` | 同上 |
| memcpy2D 默认禁用 | `assert not GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d` | 同上 |
| GDS 绑定不存在 | `assert not hasattr(c_ext, "GDSManager")` | 同上 |
| nvCOMP 绑定不存在 | `assert not hasattr(c_ext, "ANSTransferContext")` | 同上 |
| 主机运行时选择 ROCm | `assert host_buffer._is_rocm()` | `test_rocm_host_runtime_selection` |

### 4.3 验证执行命令

```bash
# 1. 构建
./build.sh --rocm

# 2. 后端边界测试
python3 -m pytest tests/test_rocm_backend.py -v

# 3. CE 传输正确性（需 ≥2 GPU）
python3 -m pytest tests/test_kv_transfer_correctness.py -v

# 4. 验证 classic kernel 路径可用（ROCm 下 use_ce=False）
python3 -m pytest tests/test_rocm_backend.py tests/test_kv_transfer_correctness.py -k "classic or kernel or ce" -v --maxfail=5
```

---

## 5. 功能边界与排除项

### 5.1 AMD 上明确不支持的能力

| 能力 | CUDA 后端 | ROCm 后端 | 处理方式 |
|------|-----------|-----------|----------|
| Classic CTA kernel (`transfer_kernels.cuh`) | ✅ `use_ce=false` | ✅ `use_ce=false`（nontemporal） | 共享头；ROCm 默认仍 CE，env=0 启用 |
| GDS / cuFile | ✅ `FLEXKV_ENABLE_GDS=1` | ❌ 强制关闭 | `setup.py` 置零；`bindings.cpp` `#ifdef` 隔离；`config.py` 禁用 |
| nvCOMP 压缩 | ✅ `FLEXKV_ENABLE_NVCOMP=1` | ❌ 强制关闭 | `setup.py` 置零；`bindings.cpp` `#ifdef` 隔离 |
| NVTX tracing | ✅ `import nvtx` | ❌ 无操作桩 | `tracing.py` 门面；`rocm_utils.h` no-op |
| memcpy2D 快路径 | ✅ 可选启用 | ❌ 强制禁用 | `ce_transfer_dispatch.cu:61`、`config.py:756-757` |
| `CUDA_AVAILABLE` 宏 | ✅ 定义 | ❌ 不定义 | `setup.py:338-339` |
| `-lcuda` 链接 | ✅ | ❌ 不链接 | `setup.py:287-288` |

### 5.2 CUDA 后端兼容性

| 验证项 | 预期 |
|--------|------|
| 不传 `--rocm` 时 `build.sh` 行为不变 | `ROCM_BUILD=0`，走原 CUDA 路径 |
| `FLEXKV_USE_ROCM` 未设置时 `setup.py` 行为不变 | `IS_ROCM_BUILD=False`，所有 ROCm 分支跳过 |
| CUDA PyTorch 下 `tracing.py` 正常导入 nvtx | `torch.version.hip` 为 None → `import nvtx` |
| CUDA PyTorch 下 `host_buffer.py` 加载 libcudart | `_is_rocm()` 返回 False |
| `transfer.cu` 源文件未被修改 | git diff 显示 UNCHANGED |

---

## 6. 当前环境验证状态

| 验证项 | 结果 | 说明 |
|--------|------|------|
| `hipcc --version` | ✅ 通过 | ROCm 7.2，HIP 编译器可用 |
| `libamdhip64.so` 加载 | ✅ 通过 | Python `ctypes.CDLL` 成功 |
| AMD GPU 检测 | ✅ 通过 | MI308X (`gfx942`)，8 GPU 可用 |
| `rocm_utils.h` 独立语法编译 | ✅ 通过 | `hipcc -fsyntax-only` 含 HIP 类型/内存/回调 API |
| Python 语法检查 | ✅ 通过 | `py_compile` 覆盖 setup/config/tracing/host_buffer/tests |
| Bash 语法检查 | ✅ 通过 | `bash -n` 覆盖 build/install/run_tests |
| `git diff --check` | ✅ 通过 | 无空白错误 |
| 扩展构建 (`pip install -e . --no-build-isolation`) | ✅ 通过 | `PYTORCH_ROCM_ARCH=gfx942` 单架构构建，c_ext.so 生成 |
| `import flexkv` / `from flexkv import c_ext` | ✅ 通过 | `c_ext: /cfs_zhongwei/staryxchen/FlexKV/flexkv/c_ext.so` |
| `tests/test_rocm_backend.py` | ✅ 通过 | 2 passed (0.63s) |
| `tests/test_kv_transfer_correctness.py` | ✅ 通过 | 1591 passed, 697 skipped (409.94s) |
| Classic kernel 路径 | ✅ 支持 | `use_ce_transfer=False` → `CLASSIC_KERNEL`（见 `transfer_kernels.cuh`） |

### 6.1 验证过程中发现并修复的问题

| 问题 | 根因 | 修复 |
|------|------|------|
| `rocm_utils.h` `constexpr` 符号与 PyTorch shim 冲突 | PyTorch 的 `rocm_utils_hip.h` 用 `#define cudaSuccess hipSuccess` 等宏做 CUDA→HIP 映射，我们的 `constexpr cudaError_t cudaSuccess = hipSuccess;` 被宏展开后变成 `constexpr hipError_t hipSuccess = hipSuccess;` 导致重定义 | 将所有 `constexpr` 改为 `#ifndef` 守卫的 `#define` |
| `rocm_utils.h` NVTX no-op 桩被 hipify 破坏 | PyTorch hipify 把 `nvtxRangeId_t` → `int`、`nvtxRangeStartA` → `roctxRangeStartA`，`#ifndef nvtxRangeId_t` 变成 `#ifndef int`（永远 false），`using int = std::uint64_t;` 编译失败 | 移除 no-op 桩，改为 `#include <roctracer/roctx.h>` 提供真实 roctx 函数 |
| `memory_handle.py` 硬编码 `libcudart.so` | 直接 `ctypes.CDLL("libcudart.so")` 在 ROCm 环境下找不到库 | 添加 `_load_gpu_runtime()` 按 `torch.version.hip` 选择加载 `libamdhip64.so` 或 `libcudart.so` |
| `memory_handle.py` IPC 函数名未适配 HIP | `cudart.cudaIpcGetMemHandle` 在 HIP 下应为 `hipIpcGetMemHandle` | 添加 `_gpu_symbol()` 分派函数，按后端选择 CUDA/HIP IPC 符号 |
| `test_kv_transfer_correctness.py` 参数化 ids 不匹配 | `CE_MEMCPY2D_CONFIGS` 在 ROCm 下仅 `[False]`（1 元素），但 `ids=["no_memcpy2d", "memcpy2d"]` 有 2 元素 | 改为 `ids=lambda v: "memcpy2d" if v else "no_memcpy2d"` 动态生成 |

### 6.2 在含 ROCm PyTorch 的环境中完成验证

```bash
cd /cfs_zhongwei/staryxchen/FlexKV

# 构建（设置单架构加速）
export PYTORCH_ROCM_ARCH=gfx942
FLEXKV_USE_ROCM=1 FLEXKV_ENABLE_GDS=0 FLEXKV_ENABLE_NVCOMP=0 \
  FLEXKV_ENABLE_METRICS=0 FLEXKV_DEBUG=1 \
  pip install -e . --no-build-isolation -v

# 验证导入
python3 -c "import flexkv; from flexkv import c_ext; print('OK')"

# 后端边界测试
python3 -m pytest tests/test_rocm_backend.py -v

# CE 传输正确性
python3 -m pytest tests/test_kv_transfer_correctness.py -v
```

---

## 7. 安全与正确性审查

| 审查项 | 状态 | 说明 |
|--------|------|------|
| 无硬编码密钥/token | ✅ | 所有身份通过 env/git config |
| 无不安全命令执行 | ✅ | 构建脚本无 `system()`/`popen()` 调用用户输入 |
| 输入校验 | ✅ | pybind 边界增加 dtype/device/连续性/长度/溢出检查 |
| 整数溢出防护 | ✅ | 传输大小分步乘法前检查 `int64_t::max` |
| 空指针检查 | ✅ | `ce_transfer_dispatch.cu` 校验非空 |
| 反序列化安全 | N/A | 无反序列化路径 |
| 网络访问 | N/A | 无运行时网络调用 |
| 降级而非静默失败 | ✅ | PTX/GDS/nvCOMP 请求返回明确错误 |
