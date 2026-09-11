#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D OVERLAPPING "energy" pool (windowed sum-of-squares), window=8,
 * stride=4 (50% overlap), with a power-of-two right-shift requantization
 * to a non-negative int8:
 *
 *   for j in [0, n):
 *     acc    = sum_{k=0}^{7} (int32)x[j*4+k] * (int32)x[j*4+k]
 *     out[j] = (int8_t) min(127, acc >> shift)
 *   (acc is always >= 0, so no lower clamp is needed.)
 *
 * window is fixed to 8, stride fixed to 4 (each input sample contributes to
 * TWO consecutive output windows). x has 4*n+4 samples (halo = window -
 * stride = 4); out has n int8 results. n is a multiple of 128. shift is a
 * runtime int (a fixed value across calls in this task, but must be READ,
 * not assumed/hardcoded).
 *
 * n is large (working set exceeds L2) and each input byte is read by TWO
 * overlapping windows, so this is DDR-bandwidth-bound with genuine re-read
 * pressure. The achievability bar streams x[] tile-by-tile via double-
 * buffered uDMA through VTCM (prefetch tile c+1 while computing tile c) to
 * hide DDR latency AND avoid re-fetching the overlapped bytes from DDR,
 * then DMAs each pooled output tile back to DDR.
 */
void candidate_kernel(const int8_t *x, int8_t *out, int n, int window, int shift);
#endif
