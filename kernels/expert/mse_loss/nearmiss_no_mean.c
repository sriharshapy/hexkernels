#include <stdint.h>
/* NEAR-MISS: sum of squared differences, omitting the 1/n mean factor.
 * Compiles fine; produces a value ~512x too large -> fails tolerance check. */
void candidate_kernel(const float *pred, const float *tgt, float *out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - tgt[i];
        acc += d * d;
    }
    /* WRONG: missing division by n */
    out[0] = acc;
}
