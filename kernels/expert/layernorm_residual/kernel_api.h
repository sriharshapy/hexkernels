#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * LayerNorm-with-residual: out = LayerNorm(x + residual), all int8.
 *
 * Fused operation:
 *   1. Add residual:  t[i] = clamp((int32)x[i] + (int32)residual[i], -128, 127)
 *   2. LayerNorm on t (pinned integer formula, SHIFT=8):
 *      a. mu    = (int32)(sum_i t[i]) / n         (truncation toward zero)
 *      b. var   = (int32)(sum_i (t[i]-mu)^2) / n  (truncation, >= 0)
 *      c. v_idx = clamp(var, 0, 255)
 *      d. inv   = inv_lut[v_idx]                  (uint8, runtime)
 *      e. d[i]  = (int32)t[i] - mu
 *      f. scaled= (d[i] * (int32)gamma[i] + 64) >> 7   (round-half-up)
 *      g. normed= (scaled * (int32)inv + 128) >> 8      (round-half-up, SHIFT=8)
 *      h. out[i]= clamp(normed + (int32)beta[i], -128, 127)
 *
 * gamma[i]: int8 per-element scale (runtime).
 * beta[i]:  int8 per-element bias  (runtime).
 * inv_lut:  256 uint8 entries, index = clamp(var, 0, 255) (runtime).
 * n=113 (NOT a multiple of 128).
 */
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
#endif /* KERNEL_API_H */
