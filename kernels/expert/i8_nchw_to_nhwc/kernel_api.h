#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * NCHW -> NHWC layout transform for C=4 channels (single batch).
 * in  is C planes, each HW contiguous bytes:  in[c*HW + p]
 * out is HW pixels, each C contiguous bytes:  out[p*C + c] = in[c*HW + p]
 *
 * With C=4 this is a 4-way interleave of the four channel planes. The four
 * planes are four concurrent input streams (stride HW apart), so a naive
 * direct-DDR gather is latency-bound; staging the planes in VTCM with a
 * double-buffered uDMA and interleaving on-chip (two levels of HVX vshuff)
 * recovers the gap. Pure data movement, no arithmetic.
 *
 * C is passed but is 4 for this task; HW is a multiple of 128.
 */
void candidate_kernel(const int8_t *in, int8_t *out, int C, int HW);
#endif
