/* Scalar fp16 fused tile MAC (tolerance-correct) — the DENOMINATOR baseline.
 * The output buffer arrives pre-loaded with an accumulator tile C0; plain
 * triple-loop float accumulate starting from the existing out[i][j] value
 * (read once per cell before it is overwritten), cast to __fp16 on store.
 * The HMX expert must beat this. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = (float)out[i*n + j];
            for (int k = 0; k < n; k++) acc += (float)A[i*n+k] * (float)B[k*n+j];
            out[i*n + j] = (hvx_hf)acc;
        }
    }
}
