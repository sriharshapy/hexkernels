/* u8_avgpool2x2_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * 2x2 non-overlapping average pool (integer truncation), stride 2, at large image
 * size so the working set (in + out ~1.3MB) exceeds L2 -> DDR-bandwidth-bound.
 * Correctness contract identical to u8_avgpool2x2. Harness owns main(); maps an
 * identity VTCM translation before the timed call. Init/reference/verify are
 * HVX-ified to keep simulated cycles within budget; the HVX reference is
 * cross-checked against a scalar golden on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef W
#define W 1024
#endif
#ifndef H
#define H 1024
#endif
#define OW (W/2)
#define OH (H/2)

static uint8_t in[W*H]    HVX_ALIGN;
static uint8_t out[OW*OH] HVX_ALIGN;
static uint8_t ref[OW*OH] HVX_ALIGN;

/* Shared HVX 2x2 truncated-average pool over `nrows` input rows (even). */
static void hvx_avgpool2x2(const uint8_t *ip, uint8_t *op, int nrows, int w) {
    int ow = w / 2;
    for (int oy = 0; oy < nrows / 2; oy++) {
        const uint8_t *r0 = ip + (2*oy)   * w;
        const uint8_t *r1 = ip + (2*oy+1) * w;
        uint8_t *o = op + oy * ow;
        int ox = 0;
        for (; ox + 128 <= ow; ox += 128) {
            HVX_Vector r0lo = *(const HVX_Vector *)(r0 + 2*ox);
            HVX_Vector r0hi = *(const HVX_Vector *)(r0 + 2*ox + 128);
            HVX_Vector r1lo = *(const HVX_Vector *)(r1 + 2*ox);
            HVX_Vector r1hi = *(const HVX_Vector *)(r1 + 2*ox + 128);
            HVX_VectorPair d0 = Q6_W_vdeal_VVR(r0hi, r0lo, -1);  /* lo=even cols, hi=odd cols */
            HVX_VectorPair d1 = Q6_W_vdeal_VVR(r1hi, r1lo, -1);
            HVX_VectorPair w0 = Q6_Wh_vadd_VubVub(Q6_V_lo_W(d0), Q6_V_hi_W(d0));  /* ub+ub -> i16 */
            HVX_VectorPair w1 = Q6_Wh_vadd_VubVub(Q6_V_lo_W(d1), Q6_V_hi_W(d1));
            HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_V_lo_W(w0), Q6_V_lo_W(w1));
            HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_V_hi_W(w0), Q6_V_hi_W(w1));
            slo = Q6_Vuh_vlsr_VuhR(slo, 2);
            shi = Q6_Vuh_vlsr_VuhR(shi, 2);
            *(HVX_Vector *)(o + ox) = Q6_Vub_vsat_VhVh(shi, slo);
        }
        for (; ox < ow; ox++) {
            unsigned a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            o[ox] = (uint8_t)((a+b+c+d)/4);
        }
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (uint8_t)(j*5+1); v_step[j] = (uint8_t)(128*5); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= W*H; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < W*H; i++) in[i] = (uint8_t)(i*5+1);
    }
    /* Edge windows (row 0). */
    in[0]=252; in[1]=253; in[W]=254; in[W+1]=255;   /* (252+253+254+255)/4=63 */
    in[2]=255; in[3]=255; in[W+2]=255; in[W+3]=255; /* all 255 -> 255 */
    in[4]=0;   in[5]=0;   in[W+4]=0;  in[W+5]=0;    /* all 0 -> 0 */
    in[6]=3;   in[7]=0;   in[W+6]=0;  in[W+7]=0;    /* 3/4 = 0 (truncate) */

    hvx_avgpool2x2(in, ref, H, W);

    {
        int bad = 0;
        for (int p = 0; p < OW*OH && !bad; p += 97) {
            int oy = p / OW, ox = p % OW;
            unsigned a=in[(2*oy)*W+2*ox], b=in[(2*oy)*W+2*ox+1], c=in[(2*oy+1)*W+2*ox], d=in[(2*oy+1)*W+2*ox+1];
            if (ref[p] != (uint8_t)((a+b+c+d)/4)) bad = 1;
        }
        for (int ox = 0; ox < 8 && !bad; ox++) {
            unsigned a=in[2*ox], b=in[2*ox+1], c=in[W+2*ox], d=in[W+2*ox+1];
            if (ref[ox] != (uint8_t)((a+b+c+d)/4)) bad = 1;
        }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= OW*OH; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < OW*OH; i++) out[i] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= OW*OH && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < OW*OH; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < OW*OH; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, OW*OH, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
