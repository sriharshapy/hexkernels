/* i8_streaming_conv_dma harness (v6, group H holdout: hvx + dma + vtcm).
 * Bandwidth-bound 5-tap FIR (correlation, asymmetric taps) + round/shift
 * requant to int8 at large N (>1.5M samples). Harness owns main(): maps VTCM
 * identity, seeds deterministic inputs (with saturation-triggering extremes),
 * computes the golden via a fast HVX reference implementation, cross-checks
 * that HVX golden against an exact scalar FIR on a dense sample (so a bug in
 * the vector golden cannot slip through), poisons out, times the candidate
 * (kernel-only pcycles), bit-exact compares. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1572864   /* 1.5M samples; x+out > 3MB >> L2 -> DDR-bandwidth-bound */
#endif
#define NTAPS 5
#define SHIFT 4
#define XLEN (N + NTAPS - 1)

static int8_t x[XLEN]   HVX_ALIGN;
static int8_t taps[NTAPS] HVX_ALIGN;
static int8_t out[N]    HVX_ALIGN;
static int8_t ref[N]    HVX_ALIGN;

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline HVX_Vector fir5_64(const int8_t *xp, const int8_t *tp) {
    HVX_Vector acc = Q6_V_vzero();
    for (int j = 0; j < NTAPS; j++) {
        HVX_Vector xb = load_ua(xp + j);
        HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
        HVX_Vector tv = Q6_Vh_vsplat_R((int32_t)(int16_t)tp[j]);
        acc = Q6_Vh_vadd_VhVh(acc, Q6_Vh_vmpyi_VhVh(xh, tv));
    }
    return acc;
}
static inline HVX_Vector requant5(HVX_Vector acc, int shift) {
    HVX_Vector vzero = Q6_V_vzero();
    int hv = shift > 0 ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_Vh_vsplat_R(hv);
    HVX_Vector absacc = Q6_Vh_vabs_Vh(acc);
    HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absacc, vhalf), shift);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, acc);
    return Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, sh), sh);
}
static int8_t fir5_ref(const int8_t *xp, const int8_t *tp, int shift) {
    int32_t acc = 0;
    for (int j = 0; j < NTAPS; j++) acc += (int32_t)xp[j] * (int32_t)tp[j];
    int32_t half = shift > 0 ? (1 << (shift - 1)) : 0;
    int32_t r = (acc >= 0) ? ((acc + half) >> shift) : -(((-acc) + half) >> shift);
    if (r > 127) r = 127; if (r < -128) r = -128;
    return (int8_t)r;
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of x[]. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*7+3); v_step[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= XLEN; i += vlen) { *(HVX_Vector *)(x + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < XLEN; i++) x[i] = (int8_t)(i*7+3);
    }
    /* Asymmetric 5-tap kernel (sum=16); NOT palindromic so conv!=correlation. */
    taps[0]=-3; taps[1]=5; taps[2]=12; taps[3]=-2; taps[4]=4;

    /* Max-magnitude / saturation-triggering values at head and tail. */
    x[0]=-128; x[1]=127; x[2]=-128; x[3]=127; x[4]=-128;
    x[XLEN-1]=127; x[XLEN-2]=-128; x[XLEN-3]=127; x[XLEN-4]=-128; x[XLEN-5]=127;

    /* Golden via HVX (fast). N is a multiple of 128 so no scalar tail here. */
    for (int i = 0; i < N; i += 128) {
        HVX_Vector lo = fir5_64(x + i,      taps);
        HVX_Vector hi = fir5_64(x + i + 64, taps);
        HVX_Vector rlo = requant5(lo, SHIFT);
        HVX_Vector rhi = requant5(hi, SHIFT);
        *(HVX_Vector *)(ref + i) = Q6_Vb_vpack_VhVh_sat(rhi, rlo);
    }

    /* Cross-check the HVX golden against an exact scalar FIR on a dense sample. */
    {
        int bad = 0;
        for (int i = 0; i < N && !bad; i++) {
            int sample = (i < 2048) || (i >= N - 2048) || ((i & 1023) == 0);
            if (!sample) continue;
            if (ref[i] != fir5_ref(x + i, taps, SHIFT)) bad = 1;
        }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, taps, out, N, NTAPS, SHIFT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
