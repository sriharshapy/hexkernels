/* i8_hardswish_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 hard-swish at large N (1B read + 1B write). Correctness
 * contract identical to i8_hardswish: relu6=clamp(x+3,0,6); out=clamp(x*relu6/6,
 * -128,127) with C truncating (toward zero) division by 6. No runtime params.
 * The HVX reference (halfword lanes for x*relu6, word lanes for the /6 magic
 * multiply, sign-split for truncation toward zero) is cross-checked against the
 * exact scalar reference on a sample. Harness owns main() and maps an identity
 * VTCM translation before the timed call. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288
#endif

static int8_t x[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

static int8_t ref_scalar(int8_t xi) {
    int v = (int)xi;
    int r6 = v + 3; if (r6 < 0) r6 = 0; else if (r6 > 6) r6 = 6;
    int prod = v * r6;
    int result = prod / 6;              /* C truncation toward zero */
    if (result > 127) result = 127;
    if (result < -128) result = -128;
    return (int8_t)result;
}

/* floor(a/6) magic (a >= 0): (a*10923) >> 16, exact for a in [0, ~64k]. */
#define HSW_MAG ((10923 & 0xFFFF) | ((10923 & 0xFFFF) << 16))

/* hard-swish over a halfword vector (64 lanes). */
static inline HVX_Vector hswish_half(HVX_Vector hv, HVX_Vector v3, HVX_Vector v0h,
                                     HVX_Vector v6, HVX_Vector vlo, HVX_Vector vhi,
                                     HVX_Vector vzeroh) {
    HVX_Vector r6 = Q6_Vh_vadd_VhVh(hv, v3);
    r6 = Q6_Vh_vmax_VhVh(r6, v0h);
    r6 = Q6_Vh_vmin_VhVh(r6, v6);
    HVX_Vector prod = Q6_Vh_vmpyi_VhVh(hv, r6);      /* fits int16, |prod| <= 768 */
    HVX_Vector absp = Q6_Vh_vabs_Vh(prod);
    HVX_VectorPair apw = Q6_Ww_vsxt_Vh(absp);        /* widen to words for the magic */
    HVX_Vector qlo = Q6_Vw_vasr_VwR(Q6_Vw_vmpyi_VwRh(Q6_V_lo_W(apw), HSW_MAG), 16);
    HVX_Vector qhi = Q6_Vw_vasr_VwR(Q6_Vw_vmpyi_VwRh(Q6_V_hi_W(apw), HSW_MAG), 16);
    HVX_Vector q16 = Q6_Vh_vasr_VwVwR_sat(qhi, qlo, 0);   /* narrow back (inverse deal) */
    HVX_Vector negq = Q6_Vh_vsub_VhVh(vzeroh, q16);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzeroh, prod); /* prod < 0 -> truncate toward zero */
    HVX_Vector res = Q6_V_vmux_QVV(neg, negq, q16);
    res = Q6_Vh_vmax_VhVh(res, vlo);
    res = Q6_Vh_vmin_VhVh(res, vhi);
    return res;
}
/* hard-swish over one input byte-vector (128 lanes). vsxt deals; the saturating
 * narrow-pack is its exact inverse, restoring natural order. */
static inline HVX_Vector hswish_bytevec(HVX_Vector xb, HVX_Vector v3, HVX_Vector v0h,
                                        HVX_Vector v6, HVX_Vector vlo, HVX_Vector vhi,
                                        HVX_Vector vzeroh) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector rlo = hswish_half(Q6_V_lo_W(xh), v3, v0h, v6, vlo, vhi, vzeroh);
    HVX_Vector rhi = hswish_half(Q6_V_hi_W(xh), v3, v0h, v6, vlo, vhi, vzeroh);
    return Q6_Vb_vasr_VhVhR_sat(rhi, rlo, 0);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*13+1); vs[j] = (int8_t)(128*13); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(x + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) x[i] = (int8_t)(i*13+1);
    }
    x[0]=-3; x[1]=-4; x[2]=-2; x[3]=-1; x[4]=0; x[5]=1; x[6]=2; x[7]=3;
    x[8]=6; x[9]=127; x[10]=-128; x[11]=-6; x[12]=4;

    HVX_Vector v3    = Q6_V_vsplat_R(0x00030003u);
    HVX_Vector v0h   = Q6_V_vzero();
    HVX_Vector v6    = Q6_V_vsplat_R(0x00060006u);
    HVX_Vector vlo   = Q6_V_vsplat_R((uint32_t)((-128 & 0xFFFF) | ((-128 & 0xFFFF) << 16)));
    HVX_Vector vhi   = Q6_V_vsplat_R(0x007F007Fu);
    HVX_Vector vzeroh= Q6_V_vzero();
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = hswish_bytevec(*(const HVX_Vector *)(x + i), v3, v0h, v6, vlo, vhi, vzeroh);
        for (; i < N; i++) ref[i] = ref_scalar(x[i]);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(x[i])) { bad = 1; break; }
        for (int i = 0; i < 13; i++) if (ref[i] != ref_scalar(x[i])) bad = 1;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N); });
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
