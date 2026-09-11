#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block sign-extend int8 -> int16 with an even/odd DEINTERLEAVED
 * output layout, n=256 (2 full 128-byte input blocks; block-granular
 * op, no scalar tail). Output has n int16 elements (widening doubles
 * the byte width per element, not the element count).
 * Semantics: for each block k (128 bytes at offset k*128 in a; 128
 * int16 at offset k*128 in out), and for j in [0, 64):
 *   out[k*128 + j]      = (int16_t)a[k*128 + 2*j]        (even bytes, sign-extended)
 *   out[k*128 + 64 + j] = (int16_t)a[k*128 + 2*j + 1]    (odd bytes, sign-extended)
 * Matches Q6_Wh_vsxt_Vb(Vu), whose low half (Q6_V_lo_W) holds the
 * even-indexed bytes sign-extended and whose high half (Q6_V_hi_W)
 * holds the odd-indexed bytes sign-extended (pinned via sim experiment
 * -- same even/odd deinterleave convention as the unsigned vzxt widen).
 */
void candidate_kernel(const int8_t *a, int16_t *out, int n);
#endif
