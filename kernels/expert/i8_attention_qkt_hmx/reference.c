/* Plain scalar int8 attention QK^T (bit-exact) -- the DENOMINATOR baseline.
 * K is already stored [S,D] row-major -- row j IS key vector j -- so
 * scores[i][j] = Q[i,:] . K[j,:] is a direct row-row dot product. The same
 * 0x40-config requant as the reference is applied per output element. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out, int S, int D) {
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Q[i*D+d] * (int)K[j*D+d];
            out[i*S + j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
    }
}
