/*
 * Compatibility shim. The real header has moved to
 * csrc/gpu_backend/nvidia/gtensor_handler.cuh during the GPU backend
 * abstraction refactor (P3). Forwarding here keeps any legacy
 * `#include "gtensor_handler.cuh"` / `#include "csrc/gtensor_handler.cuh"`
 * working without producing a second definition of BackendType /
 * GTensorHandler when the new-location header is also included.
 */
#pragma once
#include "gpu_backend/nvidia/gtensor_handler.cuh"
