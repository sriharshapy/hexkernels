#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block zero-extend uint8 -> uint16 with an even/odd DEINTERLEAVED
 * output layout, n=384 (3 full 128-byte input blocks; block-granular
 * op, no scalar tail). Output has n uint16 elements (widening doubles
 * the byte width per element, not the element COUNT).
 * Semantics: for each block k (128 bytes at offset k*128 in a; 128
 * uint16 at offset k*128 in out), and for j in [0, 64):
 *   out[k*128 + j]      = (uint16_t)a[k*128 + 2*j]        (even bytes)
 *   out[k*128 + 64 + j] = (uint16_t)a[k*128 + 2*j + 1]    (odd bytes)
 * NOTE: the output is NOT a simple sequential widen (out[i]=a[i]) --
 * it is deinterleaved into an even-half / odd-half layout, because
 * that is exactly what the HVX widening intrinsic produces.
 * Matches Q6_Wuh_vzxt_Vub(Vu), whose low half (Q6_V_lo_W) holds the
 * even-indexed bytes zero-extended and whose high half (Q6_V_hi_W)
 * holds the odd-indexed bytes zero-extended (pinned via sim experiment).
 */
void candidate_kernel(const uint8_t *a, uint16_t *out, int n);
#endif
