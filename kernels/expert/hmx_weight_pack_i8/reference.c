/* Baseline: straightforward scalar 4-deep weight pack via the decoded offset. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const int8_t *B, int8_t *out, int n) {
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            out[hvx_hmx_i8_wgt_off(k, j)] = B[k*n + j];
}
