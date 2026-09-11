#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 5-tap 1D FIR (correlation, taps NOT reversed) with fixed-point round+shift
 * requantization to int8:
 *   acc     = sum_{j=0}^{4} (int32)x[i+j] * (int32)taps[j]
 *   half    = shift > 0 ? (1 << (shift-1)) : 0
 *   r       = (acc >= 0) ? (acc + half) >> shift : -(((-acc) + half) >> shift)
 *             (round-half-away-from-zero)
 *   out[i]  = saturate_to_int8(r)                        for i in [0, n)
 *
 * x has n + ntaps - 1 samples (ntaps fixed to 5, so 4 halo samples); out has n
 * int8 results. taps[] is int8[ntaps] (runtime, do NOT hardcode). shift is a
 * runtime int (fixed value across calls in this task, but must be READ, not
 * assumed).
 *
 * n is large (working set exceeds L2), so this is DDR-bandwidth-bound. The
 * achievability bar streams x[] tile-by-tile via double-buffered uDMA through
 * VTCM (prefetch tile c+1 while computing tile c) to hide DDR latency behind
 * compute, then DMAs each output tile back to DDR.
 */
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int shift);
#endif
