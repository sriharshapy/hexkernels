#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Row-wise LayerNorm over R rows of width W=128 int8, streamed via DMA.
 * Each row is normalized independently using ITS OWN mean/variance; gamma,
 * beta (per-column, shared across rows) and inv_lut (shared across rows) are
 * runtime inputs.
 *
 * Pinned multi-step integer formula (NO floating point), per row r in [0,R):
 *   1. mu    = (int32)(sum_i x[r,i]) / W              (integer truncation toward zero)
 *   2. var   = (int32)(sum_i (x[r,i]-mu)^2) / W       (integer truncation, result >= 0)
 *   3. v_idx = (uint8) clamp(var, 0, 255)              (LUT index)
 *   4. inv   = inv_lut[v_idx]                          (uint8: proportional to 1/sqrt(var+eps),
 *                                                       runtime input, opaque to candidate)
 *   5. for i in [0,W):
 *        d[r,i]   = (int32)(x[r,i] - mu)
 *        scaled   = (d[r,i] * (int32)gamma[i] + 64) >> 7    (round-half-up)
 *        normed   = (scaled * (int32)inv + 128) >> 8        (round-half-up, SHIFT=8)
 *        out[r,i] = clamp(normed + (int32)beta[i], -128, 127)
 *
 * x, out: [R x W] int8, row-major. gamma, beta: [W] int8 (per-column, shared
 * across rows). inv_lut: 256 uint8 entries, index = clamp(var,0,255) (shared
 * across rows, but each row looks up its OWN var). W is fixed to 128.
 *
 * R is large (working set exceeds L2), so this is DDR-bandwidth-bound. The
 * achievability bar streams row-blocks of x[] via double-buffered uDMA
 * through VTCM (prefetch block c+1 while normalizing block c) to hide DDR
 * latency, then DMAs each normalized row-block back to DDR.
 */
void candidate_kernel(const int8_t *x, int8_t *out, int R, int W,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
#endif
