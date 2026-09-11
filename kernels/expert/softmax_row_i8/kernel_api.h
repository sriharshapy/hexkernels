#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Row-wise softmax over a 2D int8 matrix [R x C], producing uint8 output,
 * using a DIVISION-FREE quantized-reciprocal-LUT normalization (unlike the
 * sibling task softmax_rowwise, which normalizes via an exact integer
 * division). Softmax is applied independently over each ROW of length C.
 *
 * Two runtime LUTs (both generated ONCE by the harness with the exact
 * formulas below; the candidate must read them at runtime -- anti-hardcode):
 *   exp_lut[256]    uint8,  exp_lut[idx]    = clamp(round(255*exp((idx-255)/32.0)), 1, 255)
 *   recip_lut[256]  uint16, recip_lut[sidx] = clamp(round(65536.0/center), 1, 65535)
 *                           where center = max(1, (sidx<<6) + 32)
 *
 * Pinned multi-step integer formula (NO floating point, NO division), per
 * row r:
 *   1. m      = max(x[r*C+j], j=0..C-1)                              (int8)
 *   2. idx_j  = clamp((int)x[r*C+j] - m, -255, 0) + 255               (in [0,255])
 *   3. e_j    = exp_lut[idx_j]                                        (uint8)
 *   4. S      = sum_j (int32)e_j, j=0..C-1                            (int32)
 *   5. sidx   = clamp(S >> 6, 0, 255)                                 (SSHIFT=6)
 *   6. recip  = recip_lut[sidx]                                       (uint16)
 *   7. out[r*C+j] = (uint8_t) clamp(((int32_t)e_j*(int32_t)recip + 32768) >> 16, 0, 255)
 *
 * x:         [R x C] int8, row-major.
 * out:       [R x C] uint8, row-major.
 * exp_lut:   256 uint8 entries (runtime).
 * recip_lut: 256 uint16 entries (runtime).
 * R=8, C=100 (C is NOT a multiple of 128).
 */
void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut, const uint16_t *recip_lut);
#endif /* KERNEL_API_H */
