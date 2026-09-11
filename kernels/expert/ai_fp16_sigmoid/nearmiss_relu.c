/* Near-miss A: ReLU instead of sigmoid.
 * Fails because sigmoid(x) != max(0,x) for all x:
 *   x=0   -> sigmoid=0.5, relu=0  (differs)
 *   x<0   -> sigmoid in (0,0.5), relu=0  (differs)
 *   x>0   -> sigmoid in (0.5,1), relu=x  (differs unless x happens to equal sigmoid(x))
 * Seeded inputs include x=0 and negatives so this always differs. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        float v = (float)x[i];
        out[i] = (hvx_hf)(v > 0.0f ? v : 0.0f);   /* ReLU, not sigmoid */
    }
}
