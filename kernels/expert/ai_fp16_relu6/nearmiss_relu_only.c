#include <stdint.h>
/* NEAR-MISS A: plain ReLU — max(x, 0.0f) with NO upper bound clamp.
 * Fails for x > 6.0: passes through the value instead of clamping to 6. */
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        if (v < 0.0f) v = 0.0f;
        /* missing upper clamp: out may exceed 6.0 */
        out[i] = (hvx_hf)v;
    }
}
