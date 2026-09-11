/* i8_maxpool_relu_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Fused 2x2 max pool + ReLU on signed int8, at large image size so the working
 * set (in + out ~1.3MB) exceeds L2 -> DDR-bandwidth-bound. Correctness contract
 * identical to i8_maxpool_relu. Harness owns main(); maps an identity VTCM
 * translation before the timed call. Init/reference/verify are HVX-ified; the HVX
 * reference is cross-checked against a scalar golden on a sample.
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

static int8_t in[W*H]    HVX_ALIGN;
static int8_t out[OW*OH] HVX_ALIGN;
static int8_t ref[OW*OH] HVX_ALIGN;

/* Shared HVX signed 2x2 max pool + ReLU over `nrows` input rows (even). */
static void hvx_maxpool_relu(const int8_t *ip, int8_t *op, int nrows, int w) {
    int ow = w / 2;
    HVX_Vector vz = Q6_V_vzero();
    for (int oy = 0; oy < nrows / 2; oy++) {
        const int8_t *r0 = ip + (2*oy)   * w;
        const int8_t *r1 = ip + (2*oy+1) * w;
        int8_t *o = op + oy * ow;
        int x = 0;
        for (; x + 256 <= w; x += 256) {
            HVX_Vector a0 = *(const HVX_Vector *)(r0 + x);
            HVX_Vector a1 = *(const HVX_Vector *)(r0 + x + 128);
            HVX_Vector b0 = *(const HVX_Vector *)(r1 + x);
            HVX_Vector b1 = *(const HVX_Vector *)(r1 + x + 128);
            HVX_Vector m0 = Q6_Vb_vmax_VbVb(a0, b0);
            HVX_Vector m1 = Q6_Vb_vmax_VbVb(a1, b1);
            HVX_VectorPair p = Q6_W_vdeal_VVR(m1, m0, -1);   /* lo=even cols, hi=odd cols */
            HVX_Vector pm = Q6_Vb_vmax_VbVb(Q6_V_lo_W(p), Q6_V_hi_W(p));
            *(HVX_Vector *)(o + x/2) = Q6_Vb_vmax_VbVb(pm, vz);  /* ReLU */
        }
        for (int ox = x/2; ox < ow; ox++) {
            int a=r0[2*ox], b=r0[2*ox+1], c=r1[2*ox], d=r1[2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int mm=m>n?m:n; o[ox]=(int8_t)(mm>0?mm:0);
        }
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*7-40); v_step[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= W*H; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < W*H; i++) in[i] = (int8_t)(i*7-40);
    }
    /* Edge windows (row 0). */
    in[0]=-5;  in[1]=-10; in[W]=-3;  in[W+1]=-20;  /* all negative, max=-3 -> relu 0 */
    in[2]=-1;  in[3]=-8;  in[W+2]=-4; in[W+3]=-12; /* max=-1 -> relu 0 */
    in[4]=1;   in[5]=-50; in[W+4]=-20; in[W+5]=-30;/* max=1 -> relu 1 */
    in[6]=127; in[7]=50;  in[W+6]=60; in[W+7]=80;  /* max=127 -> 127 */
    in[8]=0;   in[9]=0;   in[W+8]=0;  in[W+9]=0;   /* all zero -> 0 */

    hvx_maxpool_relu(in, ref, H, W);

    {
        int bad = 0;
        for (int p = 0; p < OW*OH && !bad; p += 97) {
            int oy = p / OW, ox = p % OW;
            int a=in[(2*oy)*W+2*ox], b=in[(2*oy)*W+2*ox+1], c=in[(2*oy+1)*W+2*ox], d=in[(2*oy+1)*W+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int mm=m>n?m:n; int g=mm>0?mm:0;
            if (ref[p] != (int8_t)g) bad = 1;
        }
        for (int ox = 0; ox < 10 && !bad; ox++) {
            int a=in[2*ox], b=in[2*ox+1], c=in[W+2*ox], d=in[W+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int mm=m>n?m:n; int g=mm>0?mm:0;
            if (ref[ox] != (int8_t)g) bad = 1;
        }
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= OW*OH; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < OW*OH; i++) out[i] = (int8_t)0xA5;
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
