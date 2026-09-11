/* i8_prelu_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 per-channel PReLU at large N. Correctness contract
 * identical to i8_prelu (passthrough for x>0, else arithmetic-shift-scaled +
 * sat8). n_ch=8 channels, n_elem=65536 (multiple of 128 AND of the DMA tile so
 * each tile lives in a single channel). alpha[] and shift are fixed but passed
 * as args (the kernel must read them). Harness owns main(); maps an identity
 * VTCM translation before the timed call; HVX init/ref/verify keep simulated
 * cycles within the wall budget. HVX reference is cross-checked against the
 * exact scalar golden on a sample. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#define N_CH   8
#define N_ELEM 65536
#define N_TOT  (N_CH * N_ELEM)   /* 524288 */
#define SHIFT  3

static const int8_t ALPHA[N_CH] = { 1, 2, 3, 1, 4, 1, 2, 3 };

static int8_t x[N_TOT]   HVX_ALIGN;
static int8_t out[N_TOT] HVX_ALIGN;
static int8_t ref[N_TOT] HVX_ALIGN;

/* exact scalar golden (matches i8_prelu) */
static int8_t ref_scalar(int8_t v, int a, int shift) {
    if (v > 0) return v;
    int r = ((int)v * a) >> shift;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* HVX halfword PReLU for one 128-byte vector, given alpha splat (halfword). */
static inline HVX_Vector prelu_vec(HVX_Vector vx, HVX_Vector valpha, int shift, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(vx);
    HVX_Vector res[2];
    HVX_Vector parts[2]; parts[0] = Q6_V_lo_W(xh); parts[1] = Q6_V_hi_W(xh);
    for (int k = 0; k < 2; k++) {
        HVX_Vector prod    = Q6_Vh_vmpyi_VhVh(parts[k], valpha);
        HVX_Vector shifted = Q6_Vh_vasr_VhR(prod, shift);       /* arithmetic >> */
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VhVh(parts[k], vzero); /* x > 0 */
        res[k] = Q6_V_vmux_QVV(pos, parts[k], shifted);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);             /* saturating pack (sxt order) */
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of x[]. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*13+1); vs[j] = (int8_t)(128*13); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N_TOT; i += vlen) { *(HVX_Vector *)(x + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N_TOT; i++) x[i] = (int8_t)(i*13+1);
    }
    /* Boundary values in the first two channels. */
    x[0]=0; x[1]=-1; x[2]=1; x[3]=-128; x[4]=127; x[5]=-64;
    x[N_ELEM+0]=0; x[N_ELEM+1]=-1; x[N_ELEM+2]=1; x[N_ELEM+3]=-128;

    /* HVX reference, per channel (n_elem multiple of 128). */
    HVX_Vector vzero = Q6_V_vzero();
    for (int c = 0; c < N_CH; c++) {
        int a = (int)ALPHA[c];
        uint32_t aw = ((uint32_t)(a & 0xFFFF) << 16) | (a & 0xFFFF);
        HVX_Vector valpha = Q6_V_vsplat_R(aw);
        int8_t *xc = x + c*N_ELEM, *rc = ref + c*N_ELEM;
        for (int i = 0; i < N_ELEM; i += 128)
            *(HVX_Vector *)(rc + i) = prelu_vec(*(const HVX_Vector *)(xc + i), valpha, SHIFT, vzero);
    }
    /* Cross-check HVX ref against exact scalar golden on a sample. */
    {
        int bad = 0;
        for (int i = 0; i < N_TOT; i += 97) {
            int c = i / N_ELEM;
            if (ref[i] != ref_scalar(x[i], (int)ALPHA[c], SHIFT)) { bad = 1; break; }
        }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(x[i], (int)ALPHA[0], SHIFT)) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    /* Poison output (HVX). */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N_TOT; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N_TOT; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N_CH, N_ELEM, ALPHA, SHIFT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* HVX bulk verify; scalar locate on mismatch. */
    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N_TOT && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N_TOT; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < N_TOT; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N_TOT, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
