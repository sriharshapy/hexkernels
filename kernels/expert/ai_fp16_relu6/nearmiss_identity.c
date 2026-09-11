#include <stdint.h>
/* NEAR-MISS B: identity — no clamping at all.
 * Fails for x < 0 (should be 0) and x > 6 (should be 6). */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = x[i];  /* no clamp: passes all values unchanged */
    }
}
