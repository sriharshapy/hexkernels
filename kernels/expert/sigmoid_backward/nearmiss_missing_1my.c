#include <stdint.h>
/* NEAR-MISS: omits the (1 - y[i]) factor.
 * Computes dy[i]*y[i] instead of dy[i]*y[i]*(1-y[i]).
 * This is wrong by a factor of (1-y); for y=0.5 the error is 2x.
 * Compiles fine; fails tolerance check. */
void candidate_kernel(const float *y, const float *dy, float *dx, int n) {
    for (int i = 0; i < n; i++)
        /* WRONG: missing (1.0f - y[i]) multiplier */
        dx[i] = dy[i] * y[i];
}
