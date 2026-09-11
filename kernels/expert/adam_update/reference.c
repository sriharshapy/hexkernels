#include <math.h>
/* Correct scalar baseline: Adam optimizer in-place update.
 * m[i] = b1*m[i] + (1-b1)*grad[i]
 * v[i] = b2*v[i] + (1-b2)*grad[i]^2
 * mhat = m[i] / (1 - b1^t)
 * vhat = v[i] / (1 - b2^t)
 * w[i] = w[i] - lr * mhat / (sqrtf(vhat) + eps)
 */
void candidate_kernel(float *w, float *m, float *v, const float *grad, int n,
                      float lr, float b1, float b2, float eps, int t) {
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    for (int i = 0; i < n; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * grad[i];
        v[i] = b2 * v[i] + (1.0f - b2) * grad[i] * grad[i];
        float mhat = m[i] / bc1;
        float vhat = v[i] / bc2;
        w[i] = w[i] - lr * mhat / (sqrtf(vhat) + eps);
    }
}
