#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Row-wise softmax over a 2D int8 matrix of shape [R x C], producing uint8 output.
 * The softmax is applied independently over each ROW of length C.
 *
 * Pinned multi-step integer formula (same as softmax_i8, applied per row r):
 *   1. m_r    = max(x[r*C + j], j=0..C-1)
 *   2. idx_j  = clamp((int)(x[r*C+j] - m_r), -255, 0) + 255   (in [0,255])
 *   3. e_j    = exp_lut[idx_j]                                  (uint8)
 *   4. S_r    = sum of e_j over j in [0,C)                      (int32)
 *   5. out[r*C+j] = (uint8)((e_j * 255 + S_r/2) / S_r)         (round-half-down)
 *
 * x:       [R x C] int8, row-major.
 * out:     [R x C] uint8, row-major.
 * exp_lut: 256 uint8 entries (runtime).
 * R=8, C=137 (C is NOT a multiple of 128).
 */
void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut);
#endif /* KERNEL_API_H */
