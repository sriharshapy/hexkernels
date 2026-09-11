/* Near-miss A: ReLU instead of SiLU.
 * Fails because silu(x) != max(0,x): for negative x, silu(x) < 0 (not 0),
 * and for positive x, silu(x) < x (not exactly x).
 * Seeded inputs include negatives so this will always differ from the reference. */
#include <stdint.h>

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = v > 0.0f ? v : 0.0f;   /* ReLU, not SiLU */
    }
}
