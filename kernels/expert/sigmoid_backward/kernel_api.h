#ifndef KERNEL_API_H
#define KERNEL_API_H
/* Sigmoid backward pass (gradient), n=512.
 * Semantics: dx[i] = dy[i] * y[i] * (1.0f - y[i])
 * y is the sigmoid forward output in (0, 1); dy is the upstream gradient;
 * dx is the output gradient.
 * Elementwise; no reduction. Tolerance compare required (fp32 qfloat path).
 */
void candidate_kernel(const float *y, const float *dy, float *dx, int n);
#endif
