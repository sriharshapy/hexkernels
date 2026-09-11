/* Baseline: plain scalar int8 attention A.V (bit-exact), no HVX/HMX.
 *   out[i][j] = sum_k P[i*S+k] * V[k*D+j]
 * then the 0x40-config HMX requant epilogue (a plain integer-math scalar
 * helper from harness_common.h). Speedup denominator. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *P, const int8_t *V, uint16_t *out, int S, int D) {
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            int acc = 0;
            for (int k = 0; k < S; k++) acc += (int)P[i*S+k] * (int)V[k*D+j];
            out[i*D + j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
    }
}
