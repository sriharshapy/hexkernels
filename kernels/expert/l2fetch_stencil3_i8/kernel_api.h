#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* 1D 3-tap weighted stencil (blur-like), edge-replicated, saturating int8:
 *   out[i] = clamp( a[clamp(i-1,0,n-1)] + 2*a[i] + a[clamp(i+1,0,n-1)], -128, 127 )
 * for i in [0, n). At the boundaries the missing neighbor is replaced by the
 * nearest in-range element (edge replication), NOT zero-padding.
 *
 * n is large (working set exceeds L2), so the task is DDR-bandwidth-bound
 * over a SINGLE stream. To go fast, rolling-l2fetch prefetch a[] a fixed
 * distance ahead of the read pointer (continuously, not just once).
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
