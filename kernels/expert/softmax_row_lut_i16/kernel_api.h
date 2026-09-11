#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Row-wise softmax over a 2D int16 matrix of shape [R x C], producing int16
 * output. The softmax is applied independently over each ROW of length C.
 * Wider fixed-point variant of the int8/uint8 row-softmax-via-LUT drill:
 * same 256-entry LUT-index derivation (clamp the already-nonpositive
 * max-subtracted diff to [-255,0]), but both containers are int16 and the
 * output is rescaled to a 16-bit fixed-point probability in [0,32767]
 * instead of an 8-bit one in [0,255].
 *
 * Pinned multi-step integer formula (NO floating point), per row r:
 *   1. m_r    = max(x[r*C + j], j=0..C-1)                          (int16)
 *   2. idx_j  = clamp((int32_t)(x[r*C+j] - m_r), -255, 0) + 255     (in [0,255])
 *   3. e_j    = exp_lut[idx_j]                                      (uint16, runtime)
 *   4. S_r    = sum of e_j over j in [0,C)                          (int32)
 *   5. out[r*C+j] = (int16_t)(((int64_t)e_j*32767 + S_r/2) / S_r)   (round-half-down)
 *
 * x:       [R x C] int16, row-major.
 * out:     [R x C] int16, row-major.
 * exp_lut: 256 uint16 entries (runtime table; do NOT hardcode).
 * R=6, C=113 (C is NOT a multiple of 128 -- tail path).
 */
void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut);
#endif /* KERNEL_API_H */
