#pragma once

namespace tileops {

// The larger of a and b, or NaN when either is NaN, as torch.amax propagates
// it: one FMNMX.NAN (sm_80+). fmaxf, which T.max lowers to, returns the other
// operand instead.
__device__ __forceinline__ float max_nan(float a, float b) {
#if defined(__MACA_ARCH__)
  if (__builtin_isnan(a)) return a;
  if (__builtin_isnan(b)) return b;
  return fmaxf(a, b);
#else
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
#endif
}

// The smaller of a and b, or NaN when either is NaN, as torch.amin propagates
// it: one FMNMX.NAN (sm_80+).
__device__ __forceinline__ float min_nan(float a, float b) {
#if defined(__MACA_ARCH__)
  if (__builtin_isnan(a)) return a;
  if (__builtin_isnan(b)) return b;
  return fminf(a, b);
#else
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
#endif
}

}  // namespace tileops
