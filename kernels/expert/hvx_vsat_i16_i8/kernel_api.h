#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Saturating narrow int16 -> int8, n=270 (tail path: 270 = 4*64 + 14).
 *
 * `a` and `b` each have n int16 elements; `out` has 2*n int8 elements.
 * Processing happens in CHUNKS of up to 64 elements (matching one HVX
 * vector of int16). For a chunk covering source index range
 * [base, base+m) (m=64 for full chunks, m=n-base for the final partial
 * chunk) and local index j in [0, m):
 *   out[2*base + j]     = sat8(b[base + j])
 *   out[2*base + m + j] = sat8(a[base + j])
 * i.e. within each chunk, `b`'s saturated values come FIRST, `a`'s come
 * SECOND -- a CONCATENATION, not an interleave (matches HVX
 * Q6_Vb_vpack_VhVh_sat(Vu=a, Vv=b), empirically pinned: this pack
 * instruction concatenates low=Vv,high=Vu, unlike the interleaving
 * vround). sat8 clamps to [-128, 127].
 */
void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n);
#endif
