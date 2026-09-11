/* DENOMINATOR baseline -- the WHOLE SwiGLU FFN block, pure scalar (no HVX/HMX).
 * Three matmuls via plain nested-loop dot products, the gated intermediate H
 * in a plain buffer, and the SAME SiLU-LUT gate / product / requant / saturate
 * pipeline as the expert:
 *   gate = X.Wg, up = X.Wu   (two scalar projections)
 *   H    = SiLU(gate) * up   (SiLU LUT gate + elementwise product -> zero-pointed uint8)
 *   out  = H.Wd              (scalar down projection + requant + bias + saturate) */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *X, const int8_t *Wg, const int8_t *Wu,
                      const int8_t *Wd, const int32_t *bd, const uint8_t *silu_lut,
                      int8_t *out, int S, int D, int Dff) {
    static uint8_t Hu[256*256];   /* gated activation */

    /* gate = X.Wg, up = X.Wu -> scale -> SiLU gate -> product -> Hu */
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < Dff; j++) {
            int gacc = 0, uacc = 0;
            for (int k = 0; k < D; k++) {
                gacc += (int)X[i*D + k] * (int)Wg[k*Dff + j];
                uacc += (int)X[i*D + k] * (int)Wu[k*Dff + j];
            }
            int gq = sw_scale(hvx_hmx_requant_0x40(gacc), SW_SG);
            int uq = sw_scale(hvx_hmx_requant_0x40(uacc), SW_SU);
            int silu_g = (signed char)silu_lut[gq + 128];
            Hu[i*Dff + j] = sw_hu(silu_g, uq);
        }
    }

    /* out = H.Wd -> requant + bias -> saturate. */
    for (int i = 0; i < S; i++)
        for (int m = 0; m < D; m++) {
            int acc = 0;
            for (int j = 0; j < Dff; j++) acc += (int)Hu[i*Dff + j] * (int)Wd[j*D + m];
            out[i*D + m] = sw_sat_i8((sw_sx12(hvx_hmx_requant_0x40(acc)) + bd[m]) >> SW_SO);
        }
}
