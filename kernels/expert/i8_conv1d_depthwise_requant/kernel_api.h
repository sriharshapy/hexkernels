#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Per-channel (depthwise) 1D conv + requantize: int8 -> int8.
 *
 * Layout: channels-first (planar).
 *   x    : [C][L_in]  int8, L_in = L + K - 1 (VALID, no zero-pad)
 *   taps : [C][K]     int8, K taps per channel (independent filters)
 *   out  : [C][L]     int8
 *
 * Computation per output element out[c][i]:
 *   acc = sum_{k=0}^{K-1} x[c][i+k] * taps[c][k]   (int32 accumulation)
 *   Requantize (round-half-away-from-zero):
 *     v    = (int64_t)acc * mult
 *     half = shift > 0 ? (1LL << (shift-1)) : 0
 *     r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 *     r   += zp
 *     out[c][i] = saturate_to_int8(r)
 *
 * mult, shift, zp are scalar runtime params -- do NOT hardcode them.
 * C=16, L=256, K=7. Pass as runtime args for generality. */
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp);
#endif
