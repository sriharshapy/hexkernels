/* i8_moving_avg_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound box moving average (W=8 fixed) at large N. Correctness contract
 * identical to i8_moving_avg (truncate-toward-zero /W). Harness owns main(); maps
 * VTCM identity before the timed call; HVX init + HVX golden keep simulated cycles
 * in budget. The HVX golden is cross-checked against an exact SCALAR moving-average
 * on a dense sample (first/last 2048 + strided) so a vector-golden bug cannot slip. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 524288    /* in-stream + out-stream int8 = 1MB (>L2 working set) -> DDR-bound */
#endif
#define W 8
#define XLEN (N + W - 1)

static int8_t x[XLEN + 256] HVX_ALIGN;   /* +256 pad so vector loads never fault */
static int8_t out[N]  HVX_ALIGN;
static int8_t ref[N]  HVX_ALIGN;

/* --- HVX moving-average (same math as a correct candidate) for a fast golden --- */
static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline HVX_Vector unpack_lo_b(HVX_Vector xb) { return Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb)); }
static inline HVX_Vector wsum8(HVX_Vector cur, HVX_Vector nxt) {
    HVX_Vector acc = cur;
    for (int s = 1; s < 8; s++) acc = Q6_Vh_vadd_VhVh(acc, Q6_V_valign_VVR(nxt, cur, s*2));
    return acc;
}
static inline HVX_Vector div8_trunc(HVX_Vector acc) {
    HVX_Vector seven = Q6_Vh_vsplat_R(7);
    HVX_Vector sign  = Q6_Vh_vasr_VhR(acc, 15);
    HVX_Vector bias  = Q6_V_vand_VV(sign, seven);
    return Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(acc, bias), 3);
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
    /* Boundary extremes exercising truncation of negative sums. */
    x[0] = -128; x[W-1] = 127; x[XLEN-1] = -128;

    /* Golden via HVX (fast). N is a multiple of 128. */
    for (int i = 0; i < N; i += 128) {
        HVX_Vector Xa = unpack_lo_b(load_ua(x + i));
        HVX_Vector Xb = unpack_lo_b(load_ua(x + i + 64));
        HVX_Vector Xc = unpack_lo_b(load_ua(x + i + 128));
        HVX_Vector qA = div8_trunc(wsum8(Xa, Xb));
        HVX_Vector qB = div8_trunc(wsum8(Xb, Xc));
        *(HVX_Vector *)(ref + i) = Q6_Vb_vpack_VhVh_sat(qB, qA);
    }

    /* Cross-check the HVX golden against exact scalar moving average on a sample. */
    {
        int bad = 0, bi = -1;
        for (int i = 0; i < N && !bad; i++) {
            int sample = (i < 2048) || (i >= N - 2048) || ((i & 1023) == 0);
            if (!sample) continue;
            int32_t acc = 0;
            for (int j = 0; j < W; j++) acc += (int32_t)x[i + j];
            int8_t g = (int8_t)(acc / W);
            if (g != ref[i]) { bad = 1; bi = i; }
        }
        if (bad) { printf("HVXENV_INCORRECT errors=1 n=%d first_bad=%d got=%ld exp=0\n",
                          N, bi, (long)(int8_t)ref[bi]); return 2; }
    }

    /* HVX poison. */
    {
        HVX_Vector vp = Q6_Vb_vsplat_R(0xA5);
        for (int i = 0; i < N; i += 128) *(HVX_Vector *)(out + i) = vp;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, W); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* HVX bulk verify; scalar scan only on mismatch. */
    int errors = 0, fb = -1;
    {
        int mismatch = 0, i = 0;
        for (; i + 128 <= N && !mismatch; i += 128) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN;
            *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < 128; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch)
            for (int k = 0; k < N; k++) if (out[k] != ref[k]) { errors++; if (fb < 0) fb = k; }
    }
    hvx_report(errors, N, fb, fb >= 0 ? (long)(int8_t)out[fb] : 0, fb >= 0 ? (long)(int8_t)ref[fb] : 0);
    return errors ? 1 : 0;
}
