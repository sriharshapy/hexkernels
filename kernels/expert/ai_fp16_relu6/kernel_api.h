#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* FP16 ReLU6 activation: out[i] = clamp(x[i], 0.0f, 6.0f)  for n elements.
 * Semantics: (hvx_hf)(fminf(fmaxf((float)x[i], 0.0f), 6.0f))
 * Both bounds are exact fp16 constants; result is bit-exact. */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n);
#endif
