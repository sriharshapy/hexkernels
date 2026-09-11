#include <stdint.h>
/* Scalar baseline: int16 absolute value, WRAP at INT16_MIN (no saturation).
 * The (int16_t) cast of 32768 wraps to -32768, matching hardware Q6_Vh_vabs_Vh. */
void candidate_kernel(const int16_t *a, int16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int v = a[i];
        out[i] = (int16_t)(v < 0 ? -v : v);
    }
}
