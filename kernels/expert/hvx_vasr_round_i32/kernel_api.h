#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Arithmetic shift-right with rounding+saturation, int32 -> int16, n=200
 * pairs (tail path: 200 = 6*32 + 8), fixed shift amount.
 *
 * `lo` and `hi` each have n int32 elements; `out` has 2*n int16 elements,
 * INTERLEAVED (matches the real HVX vector-pair interleave of
 * Q6_Vh_vasr_VwVwR_rnd_sat(Vu=hi_vec, Vv=lo_vec, shift), which places the
 * "Vv" operand's narrowed lanes at even output positions and the "Vu"
 * operand's at odd positions):
 *   out[2*i]   = sat16( round_shift(lo[i], shift) )
 *   out[2*i+1] = sat16( round_shift(hi[i], shift) )
 * where round_shift(x, s) = (s == 0) ? x : (x + (1 << (s-1))) >> s  (arithmetic
 * shift, i.e. floor division by 2^s after adding the round-half-up bias), and
 * sat16 clamps to [-32768, 32767].
 */
void candidate_kernel(const int32_t *lo, const int32_t *hi, int16_t *out, int n, int shift);
#endif
