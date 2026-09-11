/* sol_03: HVX L2-distance-squared with unrolled 2x loop. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int32_t hsum_Vw(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out) {
    const int vlen = 128;
    HVX_Vector accA = Q6_V_vzero(), accB = Q6_V_vzero();
    int i = 0;
    for (; i + 2 * vlen <= n; i += 2 * vlen) {
        /* First vector */
        HVX_VectorPair va16_0 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(a + i));
        HVX_VectorPair vb16_0 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(b + i));
        HVX_Vector d_lo0 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(va16_0), Q6_V_lo_W(vb16_0));
        HVX_Vector d_hi0 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(va16_0), Q6_V_hi_W(vb16_0));
        HVX_VectorPair sq_lo0 = Q6_Ww_vmpy_VhVh(d_lo0, d_lo0);
        HVX_VectorPair sq_hi0 = Q6_Ww_vmpy_VhVh(d_hi0, d_hi0);
        accA = Q6_Vw_vadd_VwVw(accA, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_lo0), Q6_V_hi_W(sq_lo0)));
        accA = Q6_Vw_vadd_VwVw(accA, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_hi0), Q6_V_hi_W(sq_hi0)));
        /* Second vector */
        HVX_VectorPair va16_1 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(a + i + vlen));
        HVX_VectorPair vb16_1 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(b + i + vlen));
        HVX_Vector d_lo1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(va16_1), Q6_V_lo_W(vb16_1));
        HVX_Vector d_hi1 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(va16_1), Q6_V_hi_W(vb16_1));
        HVX_VectorPair sq_lo1 = Q6_Ww_vmpy_VhVh(d_lo1, d_lo1);
        HVX_VectorPair sq_hi1 = Q6_Ww_vmpy_VhVh(d_hi1, d_hi1);
        accB = Q6_Vw_vadd_VwVw(accB, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_lo1), Q6_V_hi_W(sq_lo1)));
        accB = Q6_Vw_vadd_VwVw(accB, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_hi1), Q6_V_hi_W(sq_hi1)));
    }
    HVX_Vector acc = Q6_Vw_vadd_VwVw(accA, accB);
    for (; i + vlen <= n; i += vlen) {
        HVX_VectorPair va16 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(a + i));
        HVX_VectorPair vb16 = Q6_Wh_vunpack_Vb(*(const HVX_Vector *)(b + i));
        HVX_Vector d_lo = Q6_Vh_vsub_VhVh(Q6_V_lo_W(va16), Q6_V_lo_W(vb16));
        HVX_Vector d_hi = Q6_Vh_vsub_VhVh(Q6_V_hi_W(va16), Q6_V_hi_W(vb16));
        HVX_VectorPair sq_lo = Q6_Ww_vmpy_VhVh(d_lo, d_lo);
        HVX_VectorPair sq_hi = Q6_Ww_vmpy_VhVh(d_hi, d_hi);
        acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_lo), Q6_V_hi_W(sq_lo)));
        acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vadd_VwVw(Q6_V_lo_W(sq_hi), Q6_V_hi_W(sq_hi)));
    }
    int32_t s = hsum_Vw(acc);
    for (; i < n; i++) {
        int32_t d = (int32_t)a[i] - (int32_t)b[i];
        s += d * d;
    }
    *out = s;
}