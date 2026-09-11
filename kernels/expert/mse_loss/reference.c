#include <stdint.h>
/* Correct scalar baseline: MSE loss.
 * Semantics: out[0] = (1.0f/n) * sum_i( (pred[i] - tgt[i])^2 ) */
void candidate_kernel(const float *pred, const float *tgt, float *out, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = pred[i] - tgt[i];
        acc += d * d;
    }
    out[0] = acc / (float)n;
}
