#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Attention score tile: Q*K^T (with 1/sqrt(d) scaling) then rowwise softmax -> uint8.
 *
 * Step A -- Scaled dot product (QK^T):
 *   For each i in [0, SEQ_Q), j in [0, SEQ_K):
 *     raw[i*SEQ_K+j] = sum_d Q[i*HEAD_DIM+d] * K[j*HEAD_DIM+d]   (int32)
 *     score32[i*SEQ_K+j] = (int32) round_half_up(raw * inv_sqrt_d, shift)
 *       where round_half_up(v, sh) = (v * inv_sqrt_d + (1<<(sh-1))) >> sh
 *
 * Step B -- Rowwise softmax (same pinned formula as softmax_i8):
 *   For each row i:
 *     Treat score32[i*SEQ_K .. i*SEQ_K+SEQ_K-1] as int8 inputs (clamped to [-128,127])
 *     1. m    = max(clamp8(score32[i*SEQ_K+j]))
 *     2. idx  = clamp((int)(clamp8(s) - m), -255, 0) + 255
 *     3. e[j] = exp_lut[idx]
 *     4. S    = sum(e[j])
 *     5. out[i*SEQ_K+j] = (uint8)((e[j]*255 + S/2) / S)
 *   where clamp8(v) = (int8_t) clamp(v, -128, 127)
 *
 * Q:         [SEQ_Q x HEAD_DIM] int8 row-major.
 * K:         [SEQ_K x HEAD_DIM] int8 row-major.
 * out:       [SEQ_Q x SEQ_K] uint8.
 * exp_lut:   256 uint8 entries (runtime, opaque).
 * inv_sqrt_d, shift: runtime fixed-point 1/sqrt(HEAD_DIM) params.
 * SEQ_Q=16, SEQ_K=16, HEAD_DIM=32.
 */
#define SEQ_Q    16
#define SEQ_K    16
#define HEAD_DIM 32
void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift);
#endif /* KERNEL_API_H */
