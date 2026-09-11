#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* FP16 type alias used throughout the HVX task suite. */
typedef __fp16 hvx_hf;

/* Pointwise Sigmoid, n=1024 elements.
 *
 * PINNED SEMANTICS (all fp32 intermediates, single final fp16 cast):
 *   sigmoid(x) = 1.0f / (1.0f + expf(-x))
 *   out[i]     = (hvx_hf)sigmoid_f32((float)x[i])
 *
 * x and out are 128-byte aligned; n=1024.
 * Handle any tail (n may not be a multiple of 128-bit HVX vector lanes).
 */
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n);
#endif
