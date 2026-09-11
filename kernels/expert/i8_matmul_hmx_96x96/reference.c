/* Pure scalar int8 96x96 matmul (bit-exact) — the DENOMINATOR baseline.
 * Naive triple-loop matmul, then the same 0x40-config requant as the
 * reference applied per output element. This is the honest scalar baseline
 * the HMX expert must beat. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, const int8_t *B, uint16_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            out[i*n + j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
    }
}
