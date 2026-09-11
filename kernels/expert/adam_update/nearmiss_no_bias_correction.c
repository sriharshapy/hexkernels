#include <math.h>
/* NEAR-MISS: omits bias correction (uses raw m and v instead of mhat/vhat).
 * This is a common implementation mistake for early timesteps where b1^t
 * and b2^t are still significantly less than 1.
 * At t=10 with b1=0.9: bc1 = 1 - 0.9^10 ~ 0.651, so mhat ~= 1.53 * m.
 * The uncorrected update will underestimate the step size.
 * Compiles fine; fails harness tolerance check.
 */
void candidate_kernel(float *w, float *m, float *v, const float *grad, int n,
                      float lr, float b1, float b2, float eps, int t) {
    (void)t;  /* bias correction step unused */
    for (int i = 0; i < n; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * grad[i];
        v[i] = b2 * v[i] + (1.0f - b2) * grad[i] * grad[i];
        /* WRONG: uses raw m and v without bias correction */
        w[i] = w[i] - lr * m[i] / (sqrtf(v[i]) + eps);
    }
}
