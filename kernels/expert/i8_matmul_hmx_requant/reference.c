/* Pure scalar int8 32x32 matmul + requantize-to-int8 (bit-exact) — the
 * DENOMINATOR baseline. Naive triple-loop matmul, then the SAME 0x40-config
 * requant narrowed to int8 as the scalar reference. This is the honest
 * baseline the HMX expert must beat. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            int r = (acc * 17 + 8) >> 4;
            out[i*n + j] = (int8_t)r;   /* exact: |r| <= 102 by input-range construction */
        }
    }
}
