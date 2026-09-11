#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention-score rescale: elementwise fixed-point rescale of raw int32
 * attention scores to int16, approximating multiplication by 1/sqrt(d) via
 * a runtime (scale_mult, scale_shift) fixed-point pair.
 *
 * raw: [M x N] int32 row-major.  out: [M x N] int16 row-major (purely
 * elementwise -- no cross-element dependency, so raw/out may equivalently
 * be treated as flat 1D arrays of M*N elements).
 *
 * Pinned formula, per element idx in [0, M*N):
 *   r    = (int64_t)raw[idx] * (int64_t)scale_mult
 *   half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q    = (r >= 0) ? ((r + half) >> scale_shift)
 *                   : -((-r + half) >> scale_shift)     (round-half-away-from-zero)
 *   out[idx] = clamp(q, -32768, 32767)
 *
 * scale_mult: int32, always POSITIVE for this task's runtime sweep (so the
 * sign of q always matches the sign of raw[idx]). scale_shift: int (>=0).
 * Both are RUNTIME parameters -- read them, do not hardcode; the harness
 * sweeps multiple (scale_mult, scale_shift) sets, including one with
 * scale_shift=0 (no rounding bias) and sets that saturate.
 *
 * M=16, N=37 (592 elements total; N=37 is NOT a multiple of 32
 * int32-lanes-per-HVX-vector -- the flattened 592-element sweep has a
 * real tail: 592 = 18*32 + 16).
 *
 * |raw[idx]| is bounded (<= 2,000,000) and scale_mult <= 500 by
 * construction, so |raw[idx]| * scale_mult stays well within int32/int64
 * range for any implementation strategy.
 */
void candidate_kernel(const int32_t *raw, int16_t *out, int M, int N,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
