/* Near-miss B: sigmoid only, forgets the *x factor.
 * Computes sigmoid(x) = 1/(1+expf(-x)) instead of silu(x) = x*sigmoid(x).
 * Fails because it omits the multiplication by x — wrong everywhere except x=1. */
#include <math.h>

void candidate_kernel(const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 1.0f / (1.0f + expf(-v));   /* sigmoid only — missing *x */
    }
}
