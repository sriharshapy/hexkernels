#ifndef KERNEL_API_H
#define KERNEL_API_H
/* SGD with momentum in-place update, n=512.
 * Semantics (in-place, both w and v are updated):
 *   v[i] = mu * v[i] + grad[i]
 *   w[i] = w[i] - lr * v[i]
 * lr=0.01f, mu=0.9f.
 */
void candidate_kernel(float *w, float *v, const float *grad, int n,
                      float lr, float mu);
#endif
