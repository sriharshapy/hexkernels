#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Batched integer LayerNorm over R independent rows of length C, int16 in
 * -> int16 out (NO floating point, NO division except the exact integer
 * mean/var truncation below). x/out are [R x C] row-major.
 *
 * UNLIKE the sibling task layernorm_row_i16 (per-COLUMN gamma[C]/beta[C]
 * shared across rows), here gamma/beta are length-R and PER-ROW: gamma[r]
 * and beta[r] are SCALAR values broadcast uniformly across all C columns
 * of row r (the affine broadcast axis is flipped). The inv-scale and
 * gamma-scale epilogue steps are also applied in the OPPOSITE order
 * (inv-scale first, gamma-scale second).
 *
 * Per row r, independently:
 *   1. mu    = (sum_c x[r][c]) / C                        (int64 sum -> int32 mu,
 *                                                          truncation toward zero)
 *   2. var   = (sum_c (x[r][c]-mu)^2) / C                  (int64 sum/accum -> int64 var,
 *                                                          truncation, >=0)
 *   3. vidx  = clamp(var >> VSHIFT, 0, 255)                 (VSHIFT=5, LUT index)
 *   4. inv   = inv_lut[vidx]                                (uint16, runtime, opaque to
 *                                                          candidate, monotonic-decreasing,
 *                                                          proportional to 1/sqrt(var))
 *   5. Per column c:
 *      a. d       = (int32)(x[r][c] - mu)
 *      b. normed  = (d * (int32)inv + 512) >> 10             (round-half-up, SHIFT2=10,
 *                                                          inv-scale applied FIRST)
 *      c. scaled  = (normed * (int32)gamma[r] + 32) >> 6      (round-half-up, SHIFT1=6,
 *                                                          gamma applied SECOND, per-row
 *                                                          SCALAR, broadcast over columns)
 *      d. out[r][c] = clamp(scaled + (int32)beta[r], -32768, 32767)  (int16 saturate)
 *
 * gamma[r], beta[r]: int16, length R (NOT length C -- per-row scalar affine,
 * broadcast across all C columns of that row; runtime, do not hardcode).
 * inv_lut: 256 uint16 entries, index = clamp(var>>5, 0, 255) (runtime, must be
 * read, not hardcoded).
 * R=6, C=90 (C NOT a multiple of 64 int16-lanes-per-HVX-vector -- real tail path).
 */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const int16_t *gamma, const int16_t *beta,
                      const uint16_t *inv_lut);
#endif /* KERNEL_API_H */
