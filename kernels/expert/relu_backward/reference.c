#include <stdint.h>
/* Correct scalar baseline: ReLU backward pass.
 * Semantics: dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f */
void candidate_kernel(const float *x, const float *dy, float *dx, int n) {
    for (int i = 0; i < n; i++)
        dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f;
}
