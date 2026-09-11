/* Near-miss A: ReLU instead of GELU.
 * Fails because gelu(x) != max(0,x) for x < 0 (gelu is slightly > 0 for
 * negative x, while relu is exactly 0).  Seeded inputs include negative
 * values so this will always differ from the reference. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        out[i] = (hvx_hf)(v > 0.0f ? v : 0.0f);   /* ReLU, not GELU */
    }
}
