#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-chunk (group) normalization over a length-n int8 vector, producing int8 output.
 *
 * The vector is split into G equal-size chunks of length chunk = n/G.
 * Each chunk is normalized independently using the same layernorm formula:
 *
 * Pinned formula (for each chunk c in [0, G)):
 *   let xc[j] = x[c*chunk + j],  j in [0, chunk)
 *
 *   1. mu_c   = (int32)(sum_j xc[j]) / chunk          (integer truncation)
 *   2. var_c  = (int32)(sum_j (xc[j]-mu_c)^2) / chunk (integer truncation)
 *   3. v_idx  = (uint8) clamp(var_c, 0, 255)
 *   4. inv_c  = inv_lut[v_idx]                         (uint8, runtime)
 *   5. For each j in [0, chunk):
 *        d[j]    = (int32)(xc[j] - mu_c)
 *        scaled  = (d[j] * (int32)gamma[c*chunk+j] + 64) >> 7
 *        normed  = (scaled * (int32)inv_c + 128) >> 8
 *        out[c*chunk+j] = (int8_t) clamp(normed + (int32)beta[c*chunk+j], -128, 127)
 *
 * gamma, beta: per-element int8 (runtime, one entry per output element).
 * inv_lut:     256 uint8 entries (runtime), index = clamp(var,0,255).
 * n, G:        runtime params; n is divisible by G.
 * n=128, G=4 -> chunk=32.
 */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
#endif /* KERNEL_API_H */
