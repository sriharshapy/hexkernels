/* Scalar int8 64x64 matmul (bit-exact) — the DENOMINATOR baseline.
 * Naive row-major triple loop (no HVX vrmpy, no HMX matrix engine), matching
 * the harness's own scalar reference exactly, followed by the same 0x40-config
 * requant epilogue. This is the naive scalar baseline the HMX expert must beat. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, const int8_t *B, uint16_t *out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            out[i*n+j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
}
