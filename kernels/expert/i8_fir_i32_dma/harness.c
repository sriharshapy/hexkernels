/* i8_fir_i32_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound 1D FIR (correlation, ntaps=8 fixed) at large N; the int32
 * output dominates memory traffic. Correctness contract identical to i8_fir_i32.
 * Harness owns main(); maps VTCM identity before the timed call; HVX init + an
 * HVX-computed golden keep simulated cycles within budget. The HVX golden is
 * cross-checked against an exact SCALAR FIR on a dense sample (first/last 2048 +
 * strided) so a bug in the vector golden cannot slip through. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 229376    /* 56 tiles x 4096; out=896KB int32 + input stream > 1MB L2 -> DDR-bound */
#endif
#define NTAPS 8
#define XLEN (N + NTAPS - 1)

static int8_t  x[XLEN]  HVX_ALIGN;
static int8_t  taps[NTAPS] HVX_ALIGN;
static int32_t out[N]   HVX_ALIGN;
static int32_t ref[N]   HVX_ALIGN;

/* --- HVX FIR (same math as a correct candidate; used to build the golden fast) --- */
static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline void fir64(const int8_t *xp, HVX_Vector *lo, HVX_Vector *hi) {
    HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    for (int j = 0; j < NTAPS; j++) {
        HVX_Vector xb = load_ua(xp + j);
        HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
        HVX_Vector tv = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[j]);
        acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, tv);
    }
    HVX_VectorPair o = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
    *lo = Q6_V_lo_W(o); *hi = Q6_V_hi_W(o);
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
    for (int j = 0; j < NTAPS; j++) taps[j] = (int8_t)(j*13 - 40);
    /* Max-magnitude products at head and tail. */
    x[0]=-128; taps[0]=-128; x[1]=127; taps[1]=-128; x[XLEN-1]=-128;

    /* Golden via HVX (fast). N is a multiple of 128 so no scalar tail here. */
    for (int i = 0; i < N; i += 128) {
        HVX_Vector lo0, hi0, lo1, hi1;
        fir64(x + i,      &lo0, &hi0);
        fir64(x + i + 64, &lo1, &hi1);
        *(HVX_Vector *)(ref + i)      = lo0;
        *(HVX_Vector *)(ref + i + 32) = hi0;
        *(HVX_Vector *)(ref + i + 64) = lo1;
        *(HVX_Vector *)(ref + i + 96) = hi1;
    }

    /* Cross-check the HVX golden against an exact scalar FIR on a dense sample. */
    {
        int bad = 0;
        for (int i = 0; i < N && !bad; i++) {
            int sample = (i < 2048) || (i >= N - 2048) || ((i & 1023) == 0);
            if (!sample) continue;
            int32_t acc = 0;
            for (int j = 0; j < NTAPS; j++) acc += (int32_t)x[i + j] * (int32_t)taps[j];
            if (acc != ref[i]) bad = 1;
        }
        if (bad) { printf("HVXENV_INCORRECT errors=1 n=%d first_bad=-1 got=0 exp=0\n", N); return 2; }
    }

    for (int i = 0; i < N; i++) out[i] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, taps, out, N, NTAPS); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
