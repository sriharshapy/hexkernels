#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Fused gather-then-add: out[i] = in_a[idx[i]] + in_b[i]
 *
 * For each output position i in [0, n):
 *   out[i] = (int8_t)( (int16_t)in_a[idx[i]] + (int16_t)in_b[i] )
 *
 * Arithmetic is two's-complement int8 wraparound (no saturation).
 * Both in_a and in_b are int8; the intermediate sum fits in int16;
 * the final result is truncated (low 8 bits) back to int8.
 *
 * Constraints:
 *   0 <= idx[i] < N_SRC  (guaranteed by harness; N_SRC=256).
 *   Indices are int32.
 *   n = 512 (NOT a multiple of 128 -- handle the tail).
 *
 * in_a: [N_SRC] int8, 128-byte aligned (source to gather from).
 * in_b: [n]     int8, 128-byte aligned (bias added element-wise).
 * idx:  [n]     int32, 128-byte aligned.
 * out:  [n]     int8, 128-byte aligned.
 */
void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                      const int32_t *idx, int8_t *out, int n);
#endif /* KERNEL_API_H */
