/* Baseline: straightforward scalar crouton pack of the int8 activation tile.
 * Zero the 2048-byte buffer, then write each A[i][k] into the HIGH byte of its
 * fp16-crouton slot via the decoded activation offset. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, uint8_t *out, int n) {
    for (int i = 0; i < 2048; i++) out[i] = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            out[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
}
