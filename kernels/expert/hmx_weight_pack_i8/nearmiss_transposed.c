/* Near-miss: packs B TRANSPOSED (reads B[j*n+k] instead of B[k*n+j]) into the
 * correct 4-deep byte layout. Compiles, fills the right bytes, but stores the
 * transposed weight -> the assembled matmul would compute A*B^T -> must FAIL. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const int8_t *B, int8_t *out, int n) {
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            out[hvx_hmx_i8_wgt_off(k, j)] = B[j*n + k];   /* transposed source */
}
