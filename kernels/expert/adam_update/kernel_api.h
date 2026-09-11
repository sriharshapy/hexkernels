#ifndef KERNEL_API_H
#define KERNEL_API_H
/* Adam optimizer in-place update, n=256.
 * Semantics (in-place, w, m, and v are all updated):
 *   m[i] = b1*m[i] + (1-b1)*grad[i]
 *   v[i] = b2*v[i] + (1-b2)*grad[i]*grad[i]
 *   mhat = m[i] / (1 - pow(b1, t))
 *   vhat = v[i] / (1 - pow(b2, t))
 *   w[i] = w[i] - lr * mhat / (sqrtf(vhat) + eps)
 * lr=0.001f, b1=0.9f, b2=0.999f, eps=1e-8f, t=10.
 */
void candidate_kernel(float *w, float *m, float *v, const float *grad, int n,
                      float lr, float b1, float b2, float eps, int t);
#endif
