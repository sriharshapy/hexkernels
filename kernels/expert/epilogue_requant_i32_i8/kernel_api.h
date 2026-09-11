#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Standalone, compute-bound (in-cache, no DMA) int32 -> int8 requantize epilogue
 * -- the natural "next step" after a GEMM tile's int32 accumulator (e.g. the
 * output of gemm_tile_i8). No zero-point (simpler than requantize_i32_i8_dma).
 *
 *   v    = (int64_t)acc[i] * (int64_t)mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)   // round half away from zero
 *   out[i] = (int8_t) clamp(r, -128, 127)
 *
 * `mult`/`shift` are runtime parameters (swept by the harness -- do not hardcode
 * them). At the pinned test magnitudes (|acc| <= 10,000,000, mult in {1,200}),
 * acc*mult stays within int32 range by construction (defensive int64 in the
 * formula above, matching the convention used elsewhere in this repo, e.g.
 * bias_add_requant/requantize_i32_i8_dma -- HVX has no native 32x32->64
 * widening multiply, so a vectorized implementation relies on this bound). */
void candidate_kernel(const int32_t *acc, int8_t *out, int n, int32_t mult, int shift);
#endif
