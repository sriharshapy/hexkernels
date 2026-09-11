/* i8_leaky_relu_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 leaky-ReLU at large N. Correctness contract identical to
 * i8_leaky_relu. A single (alpha,shift) set is passed at runtime (kept large-N so
 * the task stays DDR-bound; the kernel must still read the params). Harness owns
 * main() and maps an identity VTCM translation before the timed call. The HVX
 * reference is independently cross-checked against a scalar reference on a sample
 * (incl. injected boundaries) so a wrong vector derivation can't self-pass.
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
#define ALPHA 4
#define SHIFT 1

static int8_t x[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

/* scalar golden (matches i8_leaky_relu) */
static inline int8_t ref_scalar(int8_t xi, int alpha, int shift) {
    if (xi > 0) return xi;
    int v = ((int)xi * alpha) >> shift;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

/* HVX leaky (same math as candidate contract) for a fast reference. */
static inline HVX_Vector leaky_vec(HVX_Vector vx, HVX_Vector valpha, int shift, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(vx);
    HVX_Vector plo = Q6_Vh_vmpyi_VhVh(Q6_V_lo_W(xh), valpha);
    HVX_Vector phi = Q6_Vh_vmpyi_VhVh(Q6_V_hi_W(xh), valpha);
    HVX_Vector sat = Q6_Vb_vasr_VhVhR_sat(phi, plo, shift);
    HVX_VectorPred pos = Q6_Q_vcmp_gt_VbVb(vx, vzero);
    return Q6_V_vmux_QVV(pos, vx, sat);
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
    x[0]=0; x[1]=-1; x[2]=1; x[3]=-128; x[4]=127; x[5]=-64; x[6]=-4; x[7]=100;

    uint32_t aw = ((uint32_t)(ALPHA & 0xFFFF) << 16) | (ALPHA & 0xFFFF);
    HVX_Vector valpha = Q6_V_vsplat_R(aw);
    HVX_Vector vzero = Q6_V_vzero();
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = leaky_vec(*(const HVX_Vector *)(x + i), valpha, SHIFT, vzero);
        for (; i < N; i++) ref[i] = ref_scalar(x[i], ALPHA, SHIFT);
    }
    /* Independent scalar cross-check of the HVX reference on a sample. */
    {
        int bad = 0;
        for (int i = 0; i < N; i += 97) if (ref[i] != ref_scalar(x[i], ALPHA, SHIFT)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(x[i], ALPHA, SHIFT)) bad = 1;
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
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N, ALPHA, SHIFT); });
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
