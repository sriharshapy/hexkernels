/* i8_conv2d_tiled_bias_relu harness (v5, L3 tiled-conv composition).
 * Tiled int8 3x3 VALID conv (im2col -> matmul) + fused bias/ReLU/saturating-requant
 * epilogue.  The harness owns main(): fills seeded inputs, computes the reference by
 * im2col + HVX vrmpy (a naive scalar P*C_out*K reference would blow the sim budget),
 * cross-checks that HVX reference against an INDEPENDENT scalar golden on output
 * position p=0 (semantic guard: never trust one implementation), applies the scalar
 * epilogue, poisons output, times the candidate (kernel-only pcycles), and does a
 * bit-exact compare.  Enables the HMX context AND an identity VTCM translation so a
 * candidate may compose HMX + DMA + VTCM.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define KPAD 256   /* KK=144 padded to 2 HVX vectors for the vrmpy reference */

static uint8_t  in[C_IN*IH*IW]     HVX_ALIGN;
static int8_t   W[C_OUT*KK]        HVX_ALIGN;
static int32_t  bias[C_OUT]        HVX_ALIGN;
static uint8_t  out[C_OUT*P_OUT]   HVX_ALIGN;
static uint8_t  ref[C_OUT*P_OUT]   HVX_ALIGN;
static uint8_t  im2col[P_OUT*KPAD] HVX_ALIGN;
static int8_t   Wt[C_OUT*KPAD]     HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline uint8_t saturate_u8(int v) {
    if (v > 255) v = 255;
    if (v <   0) v = 0;
    return (uint8_t)v;
}
static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

int main(void) {
    /* Map VTCM identity so DMA-capable candidates can stage crouton tiles on-chip. */
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x51C7u;
    /* in 0..3 (positive: int8==uint8 for vrmpy), W -3..3. K=144 -> worst |acc| =
     * 144*3*3 = 1296 -> |acc*17/16| = 1377 < 2048, so the HMX 12-bit requant field
     * is exact (no saturation) and bit-exactness holds. */
    for (int i = 0; i < C_IN*IH*IW; i++) in[i] = (uint8_t)(hvx_lcg(&s) % 4);           /* 0..3 */
    for (int i = 0; i < C_OUT*KK;   i++) W[i]  = (int8_t)((int)(hvx_lcg(&s) % 7) - 3); /* -3..3 */
    for (int c = 0; c < C_OUT;      c++) bias[c] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0; /* relu-zero, uint8-sat-high, passthrough */

    /* im2col: im2col[p*KPAD + k] = in[ci][oh+kh][ow+kw], k = ci*9 + kh*3 + kw. */
    for (int i = 0; i < P_OUT*KPAD; i++) im2col[i] = 0;
    for (int i = 0; i < C_OUT*KPAD; i++) Wt[i] = 0;
    for (int p = 0; p < P_OUT; p++) {
        int oh = p / OW, ow = p % OW;
        for (int k = 0; k < KK; k++) {
            int ci = k / (KH*KW), r = k % (KH*KW), kh = r / KW, kw = r % KW;
            im2col[p*KPAD + k] = in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)];
        }
    }
    for (int co = 0; co < C_OUT; co++)
        for (int k = 0; k < KK; k++) Wt[co*KPAD + k] = W[co*KK + k];

    /* HVX vrmpy reference matmul + fused scalar epilogue (KPAD = 2 vectors). */
    const HVX_Vector zero = Q6_V_vzero();
    for (int p = 0; p < P_OUT; p++) {
        HVX_Vector va0 = *(const HVX_Vector *)(im2col + p*KPAD);
        HVX_Vector va1 = *(const HVX_Vector *)(im2col + p*KPAD + 128);
        for (int co = 0; co < C_OUT; co++) {
            HVX_Vector vb0 = *(const HVX_Vector *)(Wt + co*KPAD);
            HVX_Vector vb1 = *(const HVX_Vector *)(Wt + co*KPAD + 128);
            HVX_Vector vp = Q6_Vw_vrmpyacc_VwVbVb(zero, va0, vb0);
            vp = Q6_Vw_vrmpyacc_VwVbVb(vp, va1, vb1);
            int acc = (int)hreduce32(vp);
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            ref[co*P_OUT + p] = saturate_u8(relu);
        }
    }

    /* Independent scalar golden on output position p=0 — guards the vrmpy reference. */
    {
        int p = 0, oh = 0, ow = 0;
        for (int co = 0; co < C_OUT; co++) {
            int acc = 0;
            for (int ci = 0; ci < C_IN; ci++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        acc += (int)in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)]
                             * (int)W[co*KK + ci*(KH*KW) + kh*KW + kw];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            uint8_t g = saturate_u8(relu);
            if (g != ref[co*P_OUT + p]) {
                printf("HVXENV_REFCHECK_FAIL co=%d scalar=%d vrmpy=%d\n",
                       co, (int)g, (int)ref[co*P_OUT + p]);
                return 2;
            }
        }
    }

    for (int i = 0; i < C_OUT*P_OUT; i++) *((volatile unsigned char *)&out[i]) = 0xA5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, W, bias, out, P_OUT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < C_OUT*P_OUT; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, C_OUT*P_OUT, fb, gotv, expv);
    return errors ? 1 : 0;
}
