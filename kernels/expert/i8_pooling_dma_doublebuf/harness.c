/* i8_pooling_dma_doublebuf harness (v6, group H holdout: hvx + dma + vtcm).
 * Bandwidth-bound 1D OVERLAPPING energy pool (windowed sum-of-squares,
 * window=8, stride=4 -- 50% overlap, so each input byte is read by two
 * windows) + shift/clamp at large N (>1.5M input samples). Harness owns
 * main(): maps VTCM identity, seeds deterministic inputs (with extremes
 * that push the shift near saturation and an all-zero window), computes
 * the golden via a fast HVX reference implementation, cross-checks that
 * golden against an exact scalar formula on a dense sample, poisons out,
 * times the candidate (kernel-only pcycles), bit-exact compares. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 393216   /* output count; input = 4*N+4 ~= 1.5MB >> L2 -> DDR-bandwidth-bound */
#endif
#define XLEN (4 * N + 4)
#define SHIFT 9

static int8_t x[XLEN]   HVX_ALIGN;
static int8_t out[N]    HVX_ALIGN;
static int8_t ref[N]    HVX_ALIGN;

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static void energy_pool_hvx(const int8_t *ip, int8_t *op, int32_t *acc, int nout, int shift) {
    int j = 0;
    for (; j + 32 <= nout; j += 32) {
        HVX_Vector cur_in = *(const HVX_Vector *)(ip + (size_t)j * 4);
        HVX_Vector cur = Q6_Vw_vrmpy_VbVb(cur_in, cur_in);
        HVX_Vector nxt_in = load_ua(ip + (size_t)j * 4 + 4);
        HVX_Vector nxt = Q6_Vw_vrmpy_VbVb(nxt_in, nxt_in);
        HVX_Vector energy = Q6_Vw_vadd_VwVw(cur, nxt);
        HVX_Vector sh = Q6_Vw_vasr_VwR(energy, shift);
        *(HVX_Vector *)(acc + j) = sh;
    }
    for (; j < nout; j++) {
        int32_t a = 0;
        for (int k = 0; k < 8; k++) { int32_t v = ip[(size_t)j * 4 + k]; a += v * v; }
        acc[j] = a >> shift;
    }
    for (int jj = 0; jj < nout; jj++) {
        int32_t v = acc[jj];
        if (v > 127) v = 127;
        op[jj] = (int8_t)v;
    }
}
static int8_t pool_ref(const int8_t *xp, int shift) {
    int32_t acc = 0;
    for (int k = 0; k < 8; k++) acc += (int32_t)xp[k] * (int32_t)xp[k];
    int32_t r = acc >> shift;
    if (r > 127) r = 127;
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
    /* Extremes that push the shift near saturation (acc=8*16384=131072 -> r=256 -> clamp 127),
       and an all-zero window (acc=0 -> r=0). */
    for (int k = 0; k < 8; k++) x[k] = (int8_t)((k & 1) ? -128 : 127);              /* window 0: max energy */
    for (int k = 0; k < 8; k++) x[8 + k] = 0;                                       /* window 2: zero energy */
    for (int k = 0; k < 8; k++) x[XLEN - 8 + k] = (int8_t)((k & 1) ? 127 : -128);   /* last window: max energy */

    /* Golden via HVX (fast), processed in large chunks. */
    {
        static int32_t acc[65536] HVX_ALIGN;
        int j = 0;
        while (j < N) {
            int chunk = (N - j < 65536) ? (N - j) : 65536;
            energy_pool_hvx(x + (size_t)j * 4, ref + j, acc, chunk, SHIFT);
            j += chunk;
        }
    }

    /* Cross-check the HVX golden against an exact scalar formula on a dense sample. */
    {
        int bad = 0;
        for (int j = 0; j < N && !bad; j++) {
            int sample = (j < 256) || (j >= N - 256) || ((j & 511) == 0);
            if (!sample) continue;
            if (ref[j] != pool_ref(x + (size_t)j * 4, SHIFT)) bad = 1;
        }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, 8, SHIFT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
