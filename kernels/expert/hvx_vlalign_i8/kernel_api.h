#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-block unaligned window extraction via vlalign ("left align" --
 * the complementary shift direction from valign), n=768 (6 full
 * 128-byte blocks; block-granular op, no scalar byte tail). Fixed shift
 * amount RT=19 (a compile-time constant, not a runtime parameter).
 * Semantics: for each block index k, let combined[0:128) = b[k*128 ..
 * k*128+127] and combined[128:256) = a[k*128 .. k*128+127] (b first,
 * then a). For i in [0, 128):
 *   out[k*128 + i] = combined[128 - RT + i]
 * i.e. out[k*128+i] = a[k*128+i-RT] if i>=RT, else b[k*128+128-RT+i].
 * Matches Q6_V_vlalign_VVR(Vu=a_block, Vv=b_block, Rt=RT): the window
 * ends RT bytes before the top of {Vv:Vu} (Vv low, Vu high) -- the
 * mirror-image shift direction of valign.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
