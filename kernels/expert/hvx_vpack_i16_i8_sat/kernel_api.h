#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block int16 -> int8 pack with saturation, g=2 blocks (each of a,
 * b has 64*g=128 int16 elements; block-granular op, no scalar tail).
 * Output has 128*g=256 int8 elements.
 * Semantics: define sat8(x) = clamp(x, -128, 127). For each block k
 * (64 int16 elements at offset k*64 in a/b; 128 int8 at offset k*128
 * in out), and for i in [0, 64):
 *   out[k*128 + i]      = sat8(b[k*64 + i])
 *   out[k*128 + 64 + i] = sat8(a[k*64 + i])
 * NOTE the operand order: the FIRST 64 output bytes come from b (the
 * SECOND intrinsic argument), the LAST 64 from a (the FIRST argument)
 * -- matches Q6_Vb_vpack_VhVh_sat(Vu, Vv), whose output's low half is
 * sat(Vv) and high half is sat(Vu) (pinned via sim experiment).
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int g);
#endif
