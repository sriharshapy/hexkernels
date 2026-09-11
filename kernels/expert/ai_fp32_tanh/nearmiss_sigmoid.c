/* Near-miss A: sigmoid instead of tanh.
 * Computes sigmoid(x) = 1/(1+expf(-x)) instead of tanh(x).
 * Fails because sigmoid range is [0,1] but tanh range is [-1,1]:
 * for any negative x, sigmoid(x) < 0.5 while tanh(x) < 0 -> always differs.
 * For x=0: sigmoid(0)=0.5 but tanh(0)=0 -> differs at index 0. */
#include <math.h>

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 1.0f / (1.0f + expf(-v));   /* sigmoid, not tanh */
    }
}
