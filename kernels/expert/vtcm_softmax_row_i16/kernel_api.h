#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Row-wise softmax over a 2D int16 matrix of shape [R x C], producing int16
 * output. The softmax is applied independently over each ROW of length C.
 * Adapted from softmax_row_lut_i16's pinned 256-entry-LUT formula, but with
 * an int64 row-sum accumulator instead of int32 -- C is large enough here
 * that S_r can reach ~65535*C, which OVERFLOWS int32 (the key semantic
 * difference from the small-C softmax_row_lut_i16 task).
 *
 * Pinned multi-step integer formula (NO floating point), per row r:
 *   1. m_r    = max(x[r*C + j], j=0..C-1)                          (int16)
 *   2. idx_j  = clamp((int32_t)(x[r*C+j] - m_r), -255, 0) + 255     (in [0,255])
 *   3. e_j    = exp_lut[idx_j]                                      (uint16, runtime)
 *   4. S_r    = sum of e_j over j in [0,C)                          (int64 -- NOT int32)
 *   5. out[r*C+j] = (int16_t)(((int64_t)e_j*32767 + S_r/2) / S_r)   (round-half-down)
 *
 * x:       [R x C] int16, row-major.
 * out:     [R x C] int16, row-major.
 * exp_lut: 256 uint16 entries (runtime table; do NOT hardcode).
 * R=10, C=65500 (C is NOT a multiple of 128 -- tail path).
 *
 * The LUT lookup itself is inherently scalar (a runtime table read per
 * element -- vectorized memory gather is not safe/available in this sim).
 * The achievability bar comes from the MEMORY-ACCESS pattern: DMA each row
 * into VTCM once, do all three passes (max-scan, sum+cache, normalize) on
 * the on-chip copy, then DMA the finished row back out once -- instead of
 * repeatedly touching DDR across 3+ passes per row.
 */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut);
#endif
