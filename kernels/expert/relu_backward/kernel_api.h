#ifndef KERNEL_API_H
#define KERNEL_API_H
/* ReLU backward pass (gradient), n=512.
 * Semantics: dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f
 * x is the forward-pass input; dy is the upstream gradient; dx is the output gradient.
 * Elementwise; no reduction. Tolerance compare required (fp32 qfloat path).
 */
void candidate_kernel(const float *x, const float *dy, float *dx, int n);
#endif
