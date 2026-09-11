#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Fused embedding gather + LayerNorm: for each token i,
 *   1. Gather row: e[i, :] = table[idx[i], :]
 *   2. LayerNorm on e[i, :] (length D) -> out[i, :]
 *
 * LayerNorm uses the pinned integer formula (SHIFT=8):
 *   a. mu    = (int32)(sum_d e[d]) / D            (truncation toward zero)
 *   b. var   = (int32)(sum_d (e[d]-mu)^2) / D     (truncation, >= 0)
 *   c. v_idx = clamp(var, 0, 255)
 *   d. inv   = inv_lut[v_idx]                     (uint8, runtime; shared across tokens)
 *   e. d_val = (int32)e[j] - mu
 *   f. scaled= (d_val * (int32)gamma[j] + 64) >> 7   (round-half-up)
 *   g. normed= (scaled * (int32)inv + 128) >> 8       (round-half-up, SHIFT=8)
 *   h. out[i*D + j] = clamp(normed + (int32)beta[j], -128, 127)
 *
 * Note: each token row gets its own mu/var/inv from its own embedding values.
 *       inv_lut and gamma/beta are shared across tokens (runtime inputs).
 *
 * Parameters:
 *   T=32 tokens, D=64 embedding dim, VOCAB=256.
 *   gamma   = int8[D] per-dim scale (runtime)
 *   beta    = int8[D] per-dim bias  (runtime)
 *   inv_lut = uint8[256] (runtime)
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
#endif /* KERNEL_API_H */
