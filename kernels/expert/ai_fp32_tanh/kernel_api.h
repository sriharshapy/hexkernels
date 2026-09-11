#ifndef KERNEL_API_H
#define KERNEL_API_H
/* FP32 pointwise tanh activation, n=1024 elements.
 *
 * PINNED SEMANTICS (all fp32, no down-cast):
 *   out[i] = tanhf(x[i])
 *
 * x and out are 128-byte aligned; n=1024.
 * Handle any tail (n may not be a multiple of 128-byte HVX vector lanes).
 * #include <math.h> for tanhf.
 */
void candidate_kernel(const float *x, float *out, int n);
#endif
