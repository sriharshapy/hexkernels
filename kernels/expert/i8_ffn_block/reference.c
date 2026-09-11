/* BASELINE (denominator) — pure scalar FFN block. Both matmuls via plain
 * nested-loop dot products (no HVX/HMX matrix engine), the intermediate H
 * kept in a plain buffer, and the SAME requant/ReLU/bias/saturate pipeline
 * as the expert. */
#include "kernel_api.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      int8_t *out, int S, int D, int Dff) {
    static int8_t H[256*256];   /* intermediate activation */

    /* matmul1: acc1 = X . W1 ; H = ReLU((r1+b1)>>SH1) */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < Dff; j++) {
            int acc1 = 0;
            for (int k = 0; k < D; k++) acc1 += (int)X[i*D + k] * (int)W1[k*Dff + j];
            int r1 = ffn_sx12((acc1 * 17 + 8) >> 4);
            H[i*Dff + j] = (int8_t)ffn_relu_requant(r1 + b1[j]);
        }

    /* matmul2: acc2 = H . W2 ; out = sat_i8((r2+b2)>>SH2) */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = 0;
            for (int k = 0; k < Dff; k++) acc2 += (int)H[i*Dff + k] * (int)W2[k*D + j];
            int r2 = ffn_sx12((acc2 * 17 + 8) >> 4);
            out[i*D + j] = ffn_sat_i8((r2 + b2[j]) >> FFN_SH2);
        }
}
