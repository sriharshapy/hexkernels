/* i8_mul_requant_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 multiply+requantize at large N. Correctness contract
 * identical to i8_mul_requant (round-half-away-from-zero, sat8). A single runtime
 * (mult,shift,zp) set is passed, and inputs are bounded to [-64,63] so prod*mult
 * fits int16 halfword lanes (vectorizable); the requant still saturates in both
 * directions. The HVX reference is cross-checked against the exact int64 scalar
 * reference on a sample. Harness owns main() and maps an identity VTCM translation
 * before the timed call. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 393216   /* 3072 vectors; a+b+out = 1.15MB > 1MB L2 -> DDR-bound */
#endif
#define MULT  7
#define SHIFT 6
#define ZP    3

static int8_t a[N]   HVX_ALIGN;
static int8_t b[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

static int8_t ref_scalar(int8_t ai, int8_t bi, int32_t mult, int shift, int8_t zp) {
    int32_t prod = (int32_t)ai * (int32_t)bi;
    int64_t v = (int64_t)prod * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* HVX halfword requant (valid while prod*mult fits int16). */
static inline HVX_Vector mulrq_vec(HVX_Vector va, HVX_Vector vb,
                                   HVX_Vector vmult, HVX_Vector vhalf,
                                   HVX_Vector vzp, int shift, HVX_Vector vzero) {
    HVX_VectorPair ah = Q6_Wh_vsxt_Vb(va);
    HVX_VectorPair bh = Q6_Wh_vsxt_Vb(vb);
    HVX_Vector al[2] = { Q6_V_lo_W(ah), Q6_V_hi_W(ah) };
    HVX_Vector bl[2] = { Q6_V_lo_W(bh), Q6_V_hi_W(bh) };
    HVX_Vector res[2];
    for (int k = 0; k < 2; k++) {
        HVX_Vector prod = Q6_Vh_vmpyi_VhVh(al[k], bl[k]);       /* fits int16 */
        HVX_Vector v    = Q6_Vh_vmpyi_VhVh(prod, vmult);        /* prod*mult, fits int16 */
        HVX_Vector absv = Q6_Vh_vabs_Vh(v);
        HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
        HVX_Vector negsh= Q6_Vh_vsub_VhVh(vzero, sh);
        HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);       /* v < 0 */
        HVX_Vector r    = Q6_V_vmux_QVV(neg, negsh, sh);
        res[k] = Q6_Vh_vadd_VhVh(r, vzp);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);            /* sat8 pack, sxt order */
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fill a,b in [-64,63] via ((pat & 0x7f) - 64). */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vai[128], vbi[128], vas[128], vbs[128];
        for (int j = 0; j < 128; j++) {
            vai[j] = (int8_t)(j*5+1); vbi[j] = (int8_t)(j*9+7);
            vas[j] = (int8_t)(128*5); vbs[j] = (int8_t)(128*9);
        }
        HVX_Vector ca=*(HVX_Vector*)vai, cb=*(HVX_Vector*)vbi, sa=*(HVX_Vector*)vas, sb=*(HVX_Vector*)vbs;
        HVX_Vector m7f = Q6_V_vsplat_R(0x7f7f7f7fu), v64 = Q6_V_vsplat_R(0x40404040u);
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector *)(a + i) = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(ca, m7f), v64);
            *(HVX_Vector *)(b + i) = Q6_Vb_vsub_VbVb(Q6_V_vand_VV(cb, m7f), v64);
            ca = Q6_Vb_vadd_VbVb(ca, sa); cb = Q6_Vb_vadd_VbVb(cb, sb);
        }
        for (; i < N; i++) { a[i]=(int8_t)(((i*5+1)&0x7f)-64); b[i]=(int8_t)(((i*9+7)&0x7f)-64); }
    }
    a[0]=63;  b[0]=63;   a[1]=-64; b[1]=-64;   a[2]=0; b[2]=50;   a[3]=-64; b[3]=63;

    uint32_t mw = ((uint32_t)(MULT & 0xFFFF) << 16) | (MULT & 0xFFFF);
    int hv = (SHIFT > 0) ? (1 << (SHIFT - 1)) : 0;
    uint32_t hw = ((uint32_t)(hv & 0xFFFF) << 16) | (hv & 0xFFFF);
    uint32_t zw = ((uint32_t)(ZP & 0xFFFF) << 16) | (ZP & 0xFFFF);
    HVX_Vector vmult = Q6_V_vsplat_R(mw);
    HVX_Vector vhalf = Q6_V_vsplat_R(hw);
    HVX_Vector vzp   = Q6_V_vsplat_R(zw);
    HVX_Vector vzero = Q6_V_vzero();
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = mulrq_vec(*(const HVX_Vector *)(a + i),
                                                 *(const HVX_Vector *)(b + i),
                                                 vmult, vhalf, vzp, SHIFT, vzero);
        for (; i < N; i++) ref[i] = ref_scalar(a[i], b[i], MULT, SHIFT, ZP);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], b[i], MULT, SHIFT, ZP)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], b[i], MULT, SHIFT, ZP)) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N, (int32_t)MULT, SHIFT, (int8_t)ZP); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
