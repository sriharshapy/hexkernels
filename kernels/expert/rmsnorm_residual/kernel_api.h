#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * RMSNorm-with-residual: out = RMSNorm(x + residual), all int8.
 *
 * Fused operation:
 *   1. Add residual:  t[i] = clamp((int32)x[i] + (int32)residual[i], -128, 127)
 *   2. RMSNorm on t (pinned integer formula, SHIFT=8, NO mean subtraction):
 *      a. rms2  = (int32)(sum_i t[i]^2) / n       (truncation, >= 0)
 *      b. r_idx = clamp(rms2, 0, 255)
 *      c. inv   = inv_lut[r_idx]                   (uint8, runtime)
 *      d. scaled= (t[i] * (int32)gamma[i] + 64) >> 7  (round-half-up)
 *      e. normed= (scaled * (int32)inv + 128) >> 8     (round-half-up, SHIFT=8)
 *      f. out[i]= clamp(normed, -128, 127)
 *
 * NOTE: RMSNorm has NO mean subtraction and NO beta (additive bias).
 *
 * gamma[i]: int8 per-element scale (runtime).
 * inv_lut:  256 uint8 entries, index = clamp(rms2, 0, 255) (runtime).
 * n=113 (NOT a multiple of 128).
 */
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut);
#endif /* KERNEL_API_H */
