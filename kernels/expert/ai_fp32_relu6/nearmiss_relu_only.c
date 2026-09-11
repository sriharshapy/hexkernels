#include <math.h>
/* NEAR-MISS A: plain ReLU -- fmaxf(x, 0.0f) with NO upper bound clamp.
 * Fails for x > 6.0: passes through the value instead of clamping to 6. */
void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < 0.0f) v = 0.0f;
        /* missing upper clamp: out may exceed 6.0f */
        out[i] = v;
    }
}
