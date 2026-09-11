#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Round-and-narrow int16 -> int8 with saturation, n=210 pairs (tail path:
 * 210 = 3*64 + 18).
 *
 * `a` and `b` each have n int16 elements; `out` has 2*n int8 elements,
 * INTERLEAVED (matches the real HVX vround(Vu,Vv):sat interleave of
 * Q6_Vb_vround_VhVh_sat(Vu=a_vec, Vv=b_vec), which places the "Vv"
 * operand's rounded lanes at EVEN output positions and the "Vu" operand's
 * at ODD positions):
 *   out[2*i]   = sat8( round_div256(b[i]) )
 *   out[2*i+1] = sat8( round_div256(a[i]) )
 * where round_div256(x) = (x + 128) >> 8  (arithmetic shift, i.e. floor
 * division by 256 after adding the round-half-up bias -- ties round toward
 * +infinity, e.g. 128 -> 1, -128 -> 0), and sat8 clamps to [-128, 127].
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n);
#endif
