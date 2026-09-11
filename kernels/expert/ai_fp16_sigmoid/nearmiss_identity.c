/* Near-miss B: identity (no activation).
 * Compiles and runs but fails because the reference applies sigmoid.
 * Fails at x=0 (identity gives 0.0 but sigmoid gives 0.5) and on every
 * other element where sigmoid(x) != x (i.e., essentially all inputs). */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = x[i];   /* identity — wrong, missing sigmoid */
}
