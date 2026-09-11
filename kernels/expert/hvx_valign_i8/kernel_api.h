#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block unaligned window extraction via valign, n=1024 (8 full
 * 128-byte blocks; block-granular op, no scalar byte tail). Fixed shift
 * amount RT=37 (a compile-time constant, not a runtime parameter).
 * Semantics: for each block index k, let combined[0:128) = b[k*128 ..
 * k*128+127] and combined[128:256) = a[k*128 .. k*128+127] (b first,
 * then a). For i in [0, 128):
 *   out[k*128 + i] = combined[RT + i]
 * i.e. out[k*128+i] = b[k*128+RT+i] if RT+i<128, else a[k*128+RT+i-128].
 * Matches Q6_V_valign_VVR(Vu=a_block, Vv=b_block, Rt=RT): conceptually
 * treats {Vv:Vu} (Vv low, Vu high) as a 256-byte window and returns the
 * 128 bytes starting at offset Rt.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
