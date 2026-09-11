#include <stdint.h>
/* NEAR-MISS: gates on dy>0 instead of x>0.
 * This is the wrong condition — the ReLU gate depends on the forward input x,
 * not the upstream gradient. Compiles fine; fails whenever x and dy have
 * different signs (which the test cases ensure). */
void candidate_kernel(const float *x, const float *dy, float *dx, int n) {
    for (int i = 0; i < n; i++)
        /* WRONG: should be (x[i] > 0.0f), not (dy[i] > 0.0f) */
        dx[i] = (dy[i] > 0.0f) ? dy[i] : 0.0f;
}
