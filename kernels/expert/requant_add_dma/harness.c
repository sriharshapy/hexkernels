/* requant_add_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound residual-add requantize (int32+int32 -> int8) at large N.
 * Correctness contract identical to v4 requant_add with the requant params baked
 * (MULT=5, SHIFT=3, ZP=0). Harness owns main() and maps an identity VTCM
 * translation before the timed call so a candidate may DMA DDR<->VTCM. Inputs are
 * bounded to [-128,127] (sign-extended byte pattern) so sum*MULT stays well inside
 * int32 -> the reference is computed with HVX (fast) and cross-checked against the
 * exact int64 scalar golden on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 393216   /* 3 int32/int8 buffers: 1.5MB+1.5MB in, 384KB out -> DDR-bound */
#endif
#define MULT  5
#define SHIFT 3
#define ZP    0

static int32_t a[N]   HVX_ALIGN;
static int32_t b[N]   HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN;
static int8_t  ref[N] HVX_ALIGN;

/* exact scalar golden (matches v4 requant_add) */
static int8_t ref_scalar(int32_t ax, int32_t bx) {
    int64_t sum = (int64_t)ax + (int64_t)bx;
    int64_t v   = sum * (int64_t)MULT;
    int64_t half = (SHIFT > 0) ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r   = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* HVX word-domain requant of one word vector holding the pre-mult value. */
static inline HVX_Vector requant_word(HVX_Vector x, int32_t mult_r,
                                      HVX_Vector vhalf, int shift, HVX_Vector vzp) {
    HVX_Vector vm = Q6_Vw_vmpyi_VwRh(x, mult_r);
    HVX_Vector sg = Q6_Vw_vasr_VwR(vm, 31);
    HVX_Vector ab = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vm, sg), sg);
    HVX_Vector r  = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(ab, vhalf), shift);
    r = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(r, sg), sg);
    return Q6_Vw_vadd_VwVw(r, vzp);
}

/* Fill a[] with a bounded, varied, deterministic [-128,127] int32 pattern via HVX. */
static void fill_i32(int32_t *dst, int seed) {
    const int vlen = 128;               /* bytes per HVX vector */
    int8_t vi[128], vs[128];
    for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*seed+3); vs[j] = (int8_t)(128*seed); }
    HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
    int i = 0;
    for (; i + vlen <= N; i += vlen) {
        HVX_VectorPair h = Q6_Wh_vsxt_Vb(cur);
        HVX_VectorPair wl = Q6_Ww_vsxt_Vh(Q6_V_lo_W(h));
        HVX_VectorPair wh = Q6_Ww_vsxt_Vh(Q6_V_hi_W(h));
        *(HVX_Vector *)(dst + i)      = Q6_V_lo_W(wl);
        *(HVX_Vector *)(dst + i + 32) = Q6_V_hi_W(wl);
        *(HVX_Vector *)(dst + i + 64) = Q6_V_lo_W(wh);
        *(HVX_Vector *)(dst + i + 96) = Q6_V_hi_W(wh);
        cur = Q6_Vb_vadd_VbVb(cur, step);
    }
    for (; i < N; i++) dst[i] = (int8_t)(i*seed+3);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    fill_i32(a, 7);
    fill_i32(b, 11);
    /* edge cases: saturation both ways, rounding tie, zero */
    a[0]=127;  b[0]=127;   /* sum 254*5=1270>>3=158 -> sat +127 */
    a[1]=-128; b[1]=-128;  /* sum -256*5=-1280>>3=-160 -> sat -128 */
    a[2]=2;    b[2]=2;     /* sum 4*5=20, +4>>3=3 (tie: 20/8=2.5 -> 3) */
    a[3]=0;    b[3]=0;     /* 0 */
    a[4]=-1;   b[4]=0;     /* -5, +4 -> -(1)= ...  round half away */

    uint16_t m16 = (uint16_t)(int16_t)MULT;
    int32_t  mult_r = (int32_t)((uint32_t)m16 | ((uint32_t)m16 << 16));
    HVX_Vector vhalf = Q6_V_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_V_vsplat_R((int32_t)ZP);

    /* HVX reference */
    {
        const int vlen = 128;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            HVX_Vector s0 = Q6_Vw_vadd_VwVw(*(HVX_Vector*)(a+i),    *(HVX_Vector*)(b+i));
            HVX_Vector s1 = Q6_Vw_vadd_VwVw(*(HVX_Vector*)(a+i+32), *(HVX_Vector*)(b+i+32));
            HVX_Vector s2 = Q6_Vw_vadd_VwVw(*(HVX_Vector*)(a+i+64), *(HVX_Vector*)(b+i+64));
            HVX_Vector s3 = Q6_Vw_vadd_VwVw(*(HVX_Vector*)(a+i+96), *(HVX_Vector*)(b+i+96));
            HVX_Vector r0 = requant_word(s0, mult_r, vhalf, SHIFT, vzp);
            HVX_Vector r1 = requant_word(s1, mult_r, vhalf, SHIFT, vzp);
            HVX_Vector r2 = requant_word(s2, mult_r, vhalf, SHIFT, vzp);
            HVX_Vector r3 = requant_word(s3, mult_r, vhalf, SHIFT, vzp);
            HVX_Vector ph01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
            HVX_Vector ph23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
            *(HVX_Vector *)(ref + i) = Q6_Vb_vpack_VhVh_sat(ph23, ph01);
        }
        for (; i < N; i++) ref[i] = ref_scalar(a[i], b[i]);
    }
    /* cross-check HVX ref vs int64 scalar golden on a sample */
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], b[i])) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], b[i])) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    /* poison output */
    {
        const int vlen = 128;
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = 128;
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
