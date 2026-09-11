/* NEAR-MISS: uses channel-0's norm params (NORM_MULT[0], NORM_SHIFT[0]) for ALL
 * channels instead of the true per-channel arrays -> correct only for channel 0,
 * wrong for channels 1..15. Requant stage is otherwise exact. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define NUM_CH 16
#define MULT  5
#define SHIFT 4
#define ZP    0

#define NM0 3    /* NORM_MULT[0]  */
#define NS0 2    /* NORM_SHIFT[0] */

static inline HVX_Vector rha_hw(HVX_Vector v, HVX_Vector vhalf, int shift, HVX_Vector vzero) {
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    return Q6_V_vmux_QVV(neg, negsh, sh);
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    HVX_Vector vmult  = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf  = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp    = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vzero  = Q6_V_vzero();
    HVX_Vector vnm    = Q6_Vh_vsplat_R(NM0);
    HVX_Vector vnhalf = Q6_Vh_vsplat_R((NS0 > 0) ? (1 << (NS0 - 1)) : 0);
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_VectorPair wa = Q6_Wh_vunpack_Vb(*(const HVX_Vector*)(a + i));
        HVX_Vector r[2]; HVX_Vector parts[2] = { Q6_V_lo_W(wa), Q6_V_hi_W(wa) };
        for (int k = 0; k < 2; k++) {
            HVX_Vector nrm = rha_hw(Q6_Vh_vmpyi_VhVh(parts[k], vnm), vnhalf, NS0, vzero);
            r[k] = Q6_Vh_vadd_VhVh(rha_hw(Q6_Vh_vmpyi_VhVh(nrm, vmult), vhalf, SHIFT, vzero), vzp);
        }
        *(HVX_Vector*)(out + i) = Q6_Vb_vpack_VhVh_sat(r[1], r[0]);
    }
}
