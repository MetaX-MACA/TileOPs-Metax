#pragma once

#include <cstdint>

namespace tileops {

// MACA has no CUDA cache-policy/PTX equivalent for these helpers. Keep the
// interface and use ordinary global loads so callers remain backend agnostic.
__device__ __forceinline__ void load16(void* dst, const void* src) {
  __builtin_memcpy(dst, src, 16);
}

__device__ __forceinline__ void load16_evict_first(void* dst, const void* src) {
  load16(dst, src);
}

__device__ __forceinline__ void load16_evict_last(void* dst, const void* src) {
  load16(dst, src);
}

template <typename T>
__device__ __forceinline__ T load_evict_first(const T* src) {
  return *src;
}

template <typename T>
__device__ __forceinline__ T load_evict_last(const T* src) {
  return *src;
}

__device__ __forceinline__ float load_f32_evict_last(const float* src) {
  return *src;
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  load16(dst, src);
}

__device__ __forceinline__ void prefetch_l2(const void*) {}

}  // namespace tileops
