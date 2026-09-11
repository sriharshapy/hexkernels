#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * 2-way interleave (zip): merge two planar int8 streams into one interleaved
 * stream. For i in [0, n):
 *   out[2*i]   = a[i]
 *   out[2*i+1] = b[i]
 * `n` is the per-stream element count; `out` holds 2*n bytes. This is the
 * planar->interleaved layout transform (e.g. stereo L/R planes -> interleaved,
 * or a 2-channel NCHW->NHWC). Pure data movement, no arithmetic.
 *
 * The two concurrent input streams make a direct-DDR HVX implementation
 * latency-bound; a double-buffered DMA+VTCM implementation recovers the gap.
 */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
#endif
