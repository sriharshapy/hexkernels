#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <math.h>
/* FP32 ReLU6 activation: out[i] = fminf(fmaxf(x[i], 0.0f), 6.0f)  for n elements.
 * Both bounds are exact fp32 constants; result is bit-exact. */
void candidate_kernel(const float *x, float *out, int n);
#endif
