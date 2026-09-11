#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Per-lane conditional swap-select producing TWO outputs simultaneously,
 * n=1005 (tail path: 1005 = 7*128 + 109).
 * Semantics: let pred[i] = (a[i] > b[i]) (signed int8 compare). For
 * i in [0, n):
 *   out_hi[i] = pred[i] ? a[i] : b[i]   (the "winner" -- effectively max)
 *   out_lo[i] = pred[i] ? b[i] : a[i]   (the "loser"  -- effectively min)
 * Matches Q6_W_vswap_QVV(Qt, Vu, Vv), which returns a VectorPair whose
 * LOW half (Q6_V_lo_W) is: pred[i] ? Vu[i] : Vv[i], and whose HIGH half
 * (Q6_V_hi_W) is the complementary selection: pred[i] ? Vv[i] : Vu[i]
 * (pinned via sim experiment). With Vu=a, Vv=b this gives lo=out_lo? --
 * careful: lo(pair) = pred?a:b == out_hi as defined above, and
 * hi(pair) = pred?b:a == out_lo. So out_hi must be stored from
 * Q6_V_lo_W(pair) and out_lo from Q6_V_hi_W(pair) -- the naming looks
 * swapped versus intuition; this drill is exactly about getting that
 * mapping right.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out_hi, int8_t *out_lo, int n);
#endif
