/* NEAR-MISS: omits the fused BIAS add (treats fma = a*b instead of a*b + BIAS).
 * Requant is otherwise exact -> wrong on every element (result shifted by
 * ~BIAS*MULT/2). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define MULT  3
#define SHIFT 1
#define ZP    0

static inline HVX_Vector requant_half(HVX_Vector p, HVX_Vector vmult, HVX_Vector vhalf,
                                      int shift, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector v = Q6_Vh_vmpyi_VhVh(p, vmult);
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    HVX_Vector r = Q6_V_vmux_QVV(neg, negsh, sh);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n) {
    HVX_Vector vmult = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vzero = Q6_V_vzero();
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector a0=*(const HVX_Vector*)(a+i), a1=*(const HVX_Vector*)(a+i+32);
        HVX_Vector a2=*(const HVX_Vector*)(a+i+64), a3=*(const HVX_Vector*)(a+i+96);
        HVX_Vector b0=*(const HVX_Vector*)(b+i), b1=*(const HVX_Vector*)(b+i+32);
        HVX_Vector b2=*(const HVX_Vector*)(b+i+64), b3=*(const HVX_Vector*)(b+i+96);
        HVX_Vector ha0 = Q6_Vh_vpack_VwVw_sat(a1, a0), ha1 = Q6_Vh_vpack_VwVw_sat(a3, a2);
        HVX_Vector hb0 = Q6_Vh_vpack_VwVw_sat(b1, b0), hb1 = Q6_Vh_vpack_VwVw_sat(b3, b2);
        HVX_Vector p0 = Q6_Vh_vmpyi_VhVh(ha0, hb0);   /* no BIAS */
        HVX_Vector p1 = Q6_Vh_vmpyi_VhVh(ha1, hb1);
        HVX_Vector r0 = requant_half(p0, vmult, vhalf, SHIFT, vzp, vzero);
        HVX_Vector r1 = requant_half(p1, vmult, vhalf, SHIFT, vzp, vzero);
        *(HVX_Vector *)(out + i) = Q6_Vb_vpack_VhVh_sat(r1, r0);
    }
    for (; i < n; i++) {
        int64_t v = ((int64_t)a[i] * (int64_t)b[i]) * MULT;   /* no BIAS */
        int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
        int64_t r = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
        r += ZP; if (r > 127) r = 127; if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
