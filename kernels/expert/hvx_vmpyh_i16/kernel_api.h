#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Fractional (Q15) multiply-high with rounding+saturation, int16, n=519
 * (tail path: 519 = 8*64 + 7).
 *
 * Semantics: out[i] = sat16(round_q15((int)a[i] * (int)b[i]))  for i in
 * [0, n), where round_q15(P) = truncate_toward_zero((P + bias) / 32768)
 * with bias = +16384 if P>=0 else -16384 (i.e. round-half-away-from-zero
 * of P/32768). Matches HVX Q6_Vh_vmpy_VhVh_s1_rnd_sat, the standard Q15
 * fractional-fixed-point multiply (both operands treated as Q15 values in
 * [-1, 1); the product's "high half" is the Q15 result).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
#endif
