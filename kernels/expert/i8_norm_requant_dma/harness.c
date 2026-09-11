/* i8_norm_requant_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound per-channel normalize + requantize (int8 -> int8) at large N.
 * Inherits v4 i8_norm_requant with the per-channel arrays and requant params baked
 * (NUM_CH=16, MULT=5, SHIFT=4, ZP=0). Layout [NUM_CH][per_ch], per_ch = N/NUM_CH.
 * Harness owns main() and maps an identity VTCM translation before the timed call.
 * All intermediates fit int16 (halfword-lane compute). HVX reference cross-checked
 * against the int64 scalar golden on a sample.
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
#define NUM_CH 16
#define PER_CH (N / NUM_CH)      /* 32768, multiple of 128 */
#define MULT  5
#define SHIFT 4
#define ZP    0

static const int32_t NORM_MULT[NUM_CH]  = { 3, 5, 7, 9, 11, 13, 15, 17, 3, 5, 7, 9, 11, 13, 15, 17 };
static const int     NORM_SHIFT[NUM_CH] = { 2, 3, 4, 5,  3,  4,  5,  6, 2, 3, 4, 5,  3,  4,  5,  6 };

static int8_t a[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

static int8_t ref_scalar(int8_t ai, int32_t nm, int ns) {
    int64_t nv    = (int64_t)ai * (int64_t)nm;
    int64_t nhalf = ns > 0 ? ((int64_t)1 << (ns - 1)) : 0;
    int64_t norm  = (nv >= 0) ? ((nv + nhalf) >> ns) : -(((-nv) + nhalf) >> ns);
    int64_t v     = norm * (int64_t)MULT;
    int64_t half  = SHIFT > 0 ? ((int64_t)1 << (SHIFT - 1)) : 0;
    int64_t r     = (v >= 0) ? ((v + half) >> SHIFT) : -(((-v) + half) >> SHIFT);
    r += ZP;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static inline HVX_Vector rha_hw(HVX_Vector v, HVX_Vector vhalf, int shift, HVX_Vector vzero) {
    HVX_Vector absv = Q6_Vh_vabs_Vh(v);
    HVX_Vector sh   = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absv, vhalf), shift);
    HVX_Vector negsh = Q6_Vh_vsub_VhVh(vzero, sh);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, v);
    return Q6_V_vmux_QVV(neg, negsh, sh);
}

/* one halfword vector of activations -> requantized halfword result */
static inline HVX_Vector norm_req_hw(HVX_Vector av, HVX_Vector vnm, HVX_Vector vnhalf, int ns,
                                     HVX_Vector vmult, HVX_Vector vhalf, HVX_Vector vzp, HVX_Vector vzero) {
    HVX_Vector nrm = rha_hw(Q6_Vh_vmpyi_VhVh(av, vnm), vnhalf, ns, vzero);
    HVX_Vector r   = rha_hw(Q6_Vh_vmpyi_VhVh(nrm, vmult), vhalf, SHIFT, vzero);
    return Q6_Vh_vadd_VhVh(r, vzp);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* HVX byte fill of a[] (wraps within int8). */
    {
        const int vlen = 128;
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*13+1); vs[j] = (int8_t)(128*13); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (int8_t)(i*13+1);
    }
    a[0]=127; a[1]=-128; a[2]=0; a[3]=1; a[4]=-1;

    HVX_Vector vmult = Q6_Vh_vsplat_R(MULT);
    HVX_Vector vhalf = Q6_Vh_vsplat_R((SHIFT > 0) ? (1 << (SHIFT - 1)) : 0);
    HVX_Vector vzp   = Q6_Vh_vsplat_R(ZP);
    HVX_Vector vzero = Q6_V_vzero();

    /* HVX reference, per channel */
    for (int c = 0; c < NUM_CH; c++) {
        int32_t nm = NORM_MULT[c]; int ns = NORM_SHIFT[c];
        HVX_Vector vnm    = Q6_Vh_vsplat_R(nm);
        HVX_Vector vnhalf = Q6_Vh_vsplat_R((ns > 0) ? (1 << (ns - 1)) : 0);
        const int8_t *row = a + c * PER_CH;
        int8_t *dst = ref + c * PER_CH;
        for (int i = 0; i < PER_CH; i += 128) {
            HVX_VectorPair wa = Q6_Wh_vunpack_Vb(*(const HVX_Vector*)(row + i));
            HVX_Vector r0 = norm_req_hw(Q6_V_lo_W(wa), vnm, vnhalf, ns, vmult, vhalf, vzp, vzero);
            HVX_Vector r1 = norm_req_hw(Q6_V_hi_W(wa), vnm, vnhalf, ns, vmult, vhalf, vzp, vzero);
            *(HVX_Vector*)(dst + i) = Q6_Vb_vpack_VhVh_sat(r1, r0);
        }
    }
    /* cross-check vs scalar golden on a sample */
    {
        int bad = 0;
        for (int idx = 0; idx < N; idx += 89) {
            int c = idx / PER_CH;
            if (ref[idx] != ref_scalar(a[idx], NORM_MULT[c], NORM_SHIFT[c])) { bad = 1; break; }
        }
        for (int c = 0; c < NUM_CH; c++)
            for (int i = 0; i < 4; i++) {
                int idx = c*PER_CH + i;
                if (ref[idx] != ref_scalar(a[idx], NORM_MULT[c], NORM_SHIFT[c])) bad = 1;
            }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    {
        const int vlen = 128;
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
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
