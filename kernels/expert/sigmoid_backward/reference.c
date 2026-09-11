#include <stdint.h>
/* Correct scalar baseline: sigmoid backward pass.
 * Semantics: dx[i] = dy[i] * y[i] * (1.0f - y[i]) */
void candidate_kernel(const float *y, const float *dy, float *dx, int n) {
    for (int i = 0; i < n; i++)
        dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}
