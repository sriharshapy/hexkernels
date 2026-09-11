/* NEAR-MISS (must score INCORRECT): drops the SiLU gate. Identical HVX SwiGLU
 * flow -- two projections, elementwise product, down projection, same requant /
 * zero-point / bias / saturate -- but forgets to apply the SiLU nonlinearity to
 * the gate branch: it multiplies the RAW (linear) gate value by up instead of
 * SiLU(gate)*up. A plausible "forgot the activation LUT" bug that produces a
 * different gated activation and therefore a different output -> must FAIL. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

void candidate_kernel(const uint8_t *X, const int8_t *Wg, const int8_t *Wu,
                      const int8_t *Wd, const int32_t *bd, const uint8_t *silu_lut,
                      int8_t *out, int S, int D, int Dff) {
    static int8_t  Wgt[256*256] HVX_ALIGN;
    static int8_t  Wut[256*256] HVX_ALIGN;
    static int8_t  Wdt[256*256] HVX_ALIGN;
    static uint8_t Hu [256*256] HVX_ALIGN;

    (void)silu_lut;   /* BUG: the SiLU LUT is never applied */

    for (int k = 0; k < D;   k++) for (int j = 0; j < Dff; j++) Wgt[j*D + k]   = Wg[k*Dff + j];
    for (int k = 0; k < D;   k++) for (int j = 0; j < Dff; j++) Wut[j*D + k]   = Wu[k*Dff + j];
    for (int k = 0; k < Dff; k++) for (int m = 0; m < D;   m++) Wdt[m*Dff + k] = Wd[k*D + m];

    const HVX_VectorPred dn = Q6_Q_vsetq_R(D);
    const HVX_Vector zero = Q6_V_vzero();

    for (int i = 0; i < S; i++) {
        HVX_Vector vx = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(X + i*D), zero);
        for (int j = 0; j < Dff; j++) {
            HVX_Vector vwg = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(Wgt + j*D), zero);
            HVX_Vector vwu = Q6_V_vmux_QVV(dn, *(const HVX_UVector *)(Wut + j*D), zero);
            int gacc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vx, vwg));
            int uacc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vx, vwu));
            int gq = sw_scale(hvx_hmx_requant_0x40(gacc), SW_SG);
            int uq = sw_scale(hvx_hmx_requant_0x40(uacc), SW_SU);
            Hu[i*Dff + j] = sw_hu(gq, uq);   /* BUG: raw gq, not SiLU(gq) */
        }
    }

    for (int i = 0; i < S; i++)
        for (int m = 0; m < D; m++) {
            HVX_Vector vh = *(const HVX_UVector *)(Hu  + i*Dff);
            HVX_Vector vw = *(const HVX_UVector *)(Wdt + m*Dff);
            int acc = (int)hreduce32(Q6_Vw_vrmpyacc_VwVbVb(zero, vh, vw));
            out[i*D + m] = sw_sat_i8((sw_sx12(hvx_hmx_requant_0x40(acc)) + bd[m]) >> SW_SO);
        }
}
