/* DENOMINATOR baseline -- plain scalar C, no HVX. Token-mixing MLP + residual,
 * then channel-mixing MLP + residual, exactly per kernel_api.h. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *X,
                      const int8_t *Wt1, const int32_t *bt1,
                      const int8_t *Wt2, const int32_t *bt2,
                      const int8_t *Wc1, const int32_t *bc1,
                      const int8_t *Wc2, const int32_t *bc2,
                      int8_t *out, int S, int D, int St, int Dff) {
    static uint8_t TMh[64*64] HVX_ALIGN;
    static int8_t  Y[32*64]   HVX_ALIGN;
    static uint8_t CMh[32*64] HVX_ALIGN;

    for (int d = 0; d < D; d++) {
        for (int sp = 0; sp < St; sp++) {
            int acc = 0;
            for (int s = 0; s < S; s++) acc += (int)X[s*D+d] * (int)Wt1[sp*S+s];
            int p = (acc >> TM_SH1) + bt1[sp];
            TMh[sp*D+d] = (uint8_t)mx_relu_i8(p);
        }
    }
    for (int d = 0; d < D; d++) {
        for (int s = 0; s < S; s++) {
            int acc2 = 0;
            for (int sp = 0; sp < St; sp++) acc2 += (int)TMh[sp*D+d] * (int)Wt2[s*St+sp];
            int p2 = (acc2 >> TM_SH2) + bt2[s];
            int tmo = mx_sat_i8(p2);
            Y[s*D+d] = mx_sat_i8((int)X[s*D+d] + tmo);
        }
    }
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < Dff; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Y[s*D+d] * (int)Wc1[d*Dff+j];
            int p = (acc >> CM_SH1) + bc1[j];
            CMh[s*Dff+j] = (uint8_t)mx_relu_i8(p);
        }
        for (int d = 0; d < D; d++) {
            int acc2 = 0;
            for (int j = 0; j < Dff; j++) acc2 += (int)CMh[s*Dff+j] * (int)Wc2[j*D+d];
            int p2 = (acc2 >> CM_SH2) + bc2[d];
            int cmo = mx_sat_i8(p2);
            out[s*D+d] = mx_sat_i8((int)Y[s*D+d] + cmo);
        }
    }
}
