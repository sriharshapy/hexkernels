/* u8_maxpool2x2_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * 2x2 non-overlapping max pool, stride 2, at large image size so the working set
 * (in + out ~1.3MB) exceeds L2 -> DDR-bandwidth-bound. Correctness contract is
 * identical to u8_maxpool2x2 (window 2x2 / stride 2, fixed). Harness owns main();
 * maps an identity VTCM translation before the timed call so a candidate may DMA
 * DDR<->VTCM. Init/reference/verify are HVX-ified to keep simulated cycles inside
 * the sim wall-clock budget; the HVX reference is cross-checked against a scalar
 * golden on a sample.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef W
#define W 1024            /* input width  (mult of 256 -> clean HVX, OW mult 128) */
#endif
#ifndef H
#define H 1024            /* input height (even) */
#endif
#define OW (W/2)
#define OH (H/2)

static uint8_t in[W*H]    HVX_ALIGN;
static uint8_t out[OW*OH] HVX_ALIGN;
static uint8_t ref[OW*OH] HVX_ALIGN;

/* Shared HVX 2x2 max-pool over a contiguous block of `nrows` input rows (even),
 * producing nrows/2 output rows. Used for the harness reference. */
static void hvx_maxpool2x2(const uint8_t *ip, uint8_t *op, int nrows, int w) {
    int ow = w / 2;
    for (int oy = 0; oy < nrows / 2; oy++) {
        const uint8_t *r0 = ip + (2*oy)   * w;
        const uint8_t *r1 = ip + (2*oy+1) * w;
        uint8_t *o = op + oy * ow;
        int x = 0;
        for (; x + 256 <= w; x += 256) {
            HVX_Vector a0 = *(const HVX_Vector *)(r0 + x);
            HVX_Vector a1 = *(const HVX_Vector *)(r0 + x + 128);
            HVX_Vector b0 = *(const HVX_Vector *)(r1 + x);
            HVX_Vector b1 = *(const HVX_Vector *)(r1 + x + 128);
            HVX_Vector m0 = Q6_Vub_vmax_VubVub(a0, b0);
            HVX_Vector m1 = Q6_Vub_vmax_VubVub(a1, b1);
            HVX_VectorPair p = Q6_W_vdeal_VVR(m1, m0, -1);   /* lo=even cols, hi=odd cols */
            *(HVX_Vector *)(o + x/2) = Q6_Vub_vmax_VubVub(Q6_V_lo_W(p), Q6_V_hi_W(p));
        }
        for (int ox = x/2; ox < ow; ox++) {
            int a = r0[2*ox], b = r0[2*ox+1], c = r1[2*ox], d = r1[2*ox+1];
            int m = a > b ? a : b; int n = c > d ? c : d; o[ox] = (uint8_t)(m > n ? m : n);
        }
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of in[]. */
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
    in[0]=255; in[1]=10;  in[W]=20;  in[W+1]=254;    /* max 255 */
    in[2]=0;   in[3]=0;   in[W+2]=0; in[W+3]=0;      /* all zero -> 0 */
    in[4]=100; in[5]=100; in[W+4]=100; in[W+5]=100;  /* all same -> 100 */

    hvx_maxpool2x2(in, ref, H, W);

    /* Cross-check HVX reference against a scalar golden on a sample. */
    {
        int bad = 0;
        for (int p = 0; p < OW*OH && !bad; p += 97) {
            int oy = p / OW, ox = p % OW;
            int a=in[(2*oy)*W+2*ox], b=in[(2*oy)*W+2*ox+1], c=in[(2*oy+1)*W+2*ox], d=in[(2*oy+1)*W+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int g=m>n?m:n;
            if (ref[p] != (uint8_t)g) bad = 1;
        }
        for (int ox = 0; ox < 6 && !bad; ox++) {
            int a=in[2*ox], b=in[2*ox+1], c=in[W+2*ox], d=in[W+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int g=m>n?m:n;
            if (ref[ox] != (uint8_t)g) bad = 1;
        }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }

    /* Poison output (HVX). */
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

    /* HVX bulk verify; scalar scan only on mismatch. */
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
