#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* Fixed-point scale-by-1.5 with round-to-nearest, saturating to int8:
 *   out[i] = clamp( (a[i]*3 + 1) >> 1, -128, 127 )   for i in [0, n)
 * The ">>1" is an ARITHMETIC (sign-extending) shift, i.e. floor(t/2) for the
 * intermediate t = a[i]*3+1 (matches C's ">>" on signed ints on this
 * toolchain and Q6_Vh_vasr_VhR on HVX).
 *
 * n is large (working set exceeds L2), so the task is DDR-bandwidth-bound
 * over a SINGLE stream. To go fast, rolling-l2fetch prefetch a[] a fixed
 * distance ahead of the read pointer (continuously, not just once) and
 * unroll the compute so more loads are in flight per prefetch.
 */
void candidate_kernel(const int8_t *a, int8_t *out, int n);
#endif
