/*
 * NVTX include-path shim for ROCm.
 *
 * torch's hipify tool automatically rewrites NVTX API calls used in this
 * codebase (nvtxRangeStartA/nvtxRangeEnd/nvtxRangePushA/nvtxRangePop/
 * nvtxRangeId_t, ...) to their roctx equivalents at the call site -- no
 * manual code changes are needed for that part.
 *
 * The one thing hipify gets wrong is the *header path*: it textually
 * replaces the "nvToolsExt.h" substring with "roctracer/roctx.h" but leaves
 * the surrounding "nvtx3/" prefix in place, producing an invalid
 * "nvtx3/roctracer/roctx.h" path. This header sidesteps that by selecting
 * the right, unmangled include path directly based on the platform macro
 * (defined via compiler flags by torch's HIP build, independent of header
 * inclusion order), and lets hipify's identifier-level rewriting handle the
 * rest as usual.
 */
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#include <roctracer/roctx.h>
#else
#include <nvtx3/nvToolsExt.h>
#endif
