/* i8_element_affine_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 affine-requantize at large N. Correctness contract
 * identical to i8_element_affine (round-half-away-from-zero, sat8). A single
 * runtime (scale,shift,s) set is passed, chosen so a[i]*scale+shift stays in
 * int16 range (vectorizable in halfword lanes) and both saturation directions
 * are exercised; the kernel must still read the params. Harness owns main() and
 * maps an identity VTCM translation before the timed call. The HVX reference is
 * cross-checked against the exact int64 scalar reference on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288
#endif
#define SCALE 5
#define SHIFT 8
#define SS    2

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

/* exact scalar golden (matches i8_element_affine) */
static int8_t ref_scalar(int8_t ai, int32_t scale, int32_t shift, int s) {
    int64_t v = (int64_t)ai * (int64_t)scale + (int64_t)shift;
    int64_t half = (s > 0) ? ((int64_t)1 << (s - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> s) : -(((-v) + half) >> s);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* HVX halfword affine (valid while a*scale+shift fits int16). */
static inline HVX_Vector affine_vec(HVX_Vector va, HVX_Vector vscale, HVX_Vector vshift,
                                    HVX_Vector vhalf, int s, HVX_Vector vzero) {
    HVX_VectorPair ah = Q6_Wh_vsxt_Vb(va);
    HVX_Vector res[2];
    HVX_Vector parts[2]; parts[0] = Q6_V_lo_W(ah); parts[1] = Q6_V_hi_W(ah);
    for (int k = 0; k < 2; k++) {
        HVX_Vector v = Q6_Vh_vadd_VhVh(Q6_Vh_vmpyi_VhVh(parts[k], vscale), vshift);
        HVX_Vector absv = Q6_Vh_vabs_Vh(v);
        HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), s);
        HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
        HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);   /* v < 0 */
        res[k] = Q6_V_vmux_QVV(neg, negsh, sh);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);   /* shift-0 = saturating pack, sxt-order */
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
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (int8_t)(i*13+1);
    }
    a[0]=-128; a[1]=127; a[2]=0; a[3]=-1; a[4]=100; a[5]=-100;

    uint32_t sw = ((uint32_t)(SCALE & 0xFFFF) << 16) | (SCALE & 0xFFFF);
    uint32_t hw = ((uint32_t)(SHIFT & 0xFFFF) << 16) | (SHIFT & 0xFFFF);
    int hv = (SS > 0) ? (1 << (SS - 1)) : 0;
    uint32_t hlfw = ((uint32_t)(hv & 0xFFFF) << 16) | (hv & 0xFFFF);
    HVX_Vector vscale = Q6_V_vsplat_R(sw);
    HVX_Vector vshift = Q6_V_vsplat_R(hw);
    HVX_Vector vhalf  = Q6_V_vsplat_R(hlfw);
    HVX_Vector vzero  = Q6_V_vzero();
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = affine_vec(*(const HVX_Vector *)(a + i),
                                                  vscale, vshift, vhalf, SS, vzero);
        for (; i < N; i++) ref[i] = ref_scalar(a[i], SCALE, SHIFT, SS);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], SCALE, SHIFT, SS)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], SCALE, SHIFT, SS)) bad = 1;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, SCALE, SHIFT, SS); });
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
