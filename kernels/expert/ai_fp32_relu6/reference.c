#include <math.h>
/* Correct scalar baseline: ReLU6 clamping in fp32.
 * out[i] = fminf(fmaxf(x[i], 0.0f), 6.0f) */
void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        out[i] = v;
    }
}
