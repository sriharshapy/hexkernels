/* i8_depthwise_separable_tiled harness (v5, L3 depthwise-separable composition).
 * 3x3 depthwise (per-channel) + 1x1 pointwise (matmul) + fused bias/ReLU/saturating
 * requant.  The harness owns main(): fills seeded inputs, computes the reference
 * (scalar depthwise -- cheap; then the pointwise via HVX vrmpy so a naive P*C_out*C_in
 * reference stays in the sim budget), cross-checks the vrmpy pointwise against an
 * INDEPENDENT scalar golden on output position p=0, poisons output, times the candidate
 * (kernel-only pcycles), and does a bit-exact compare.  Enables the HMX context AND an
 * identity VTCM translation so a candidate may compose HVX + HMX + DMA + VTCM.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define KPAD 128   /* C_in=64 padded to 1 HVX vector for the vrmpy pointwise reference */

static uint8_t  in[IH*IW*C_IN]     HVX_ALIGN;
static int8_t   Wdw[C_IN*KH*KW]    HVX_ALIGN;
static int8_t   Wpw[C_OUT*C_IN]    HVX_ALIGN;
static int32_t  bias[C_OUT]        HVX_ALIGN;
static uint8_t  out[P_OUT*C_OUT]   HVX_ALIGN;
static uint8_t  ref[P_OUT*C_OUT]   HVX_ALIGN;
static uint8_t  dw[P_OUT*C_IN]     HVX_ALIGN;   /* depthwise result, 0..27 */
static uint8_t  act8[P_OUT*KPAD]   HVX_ALIGN;   /* padded activation for vrmpy pointwise */
static int8_t   Wt8[C_OUT*KPAD]    HVX_ALIGN;   /* padded pointwise weights */

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
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x77A1u;
    /* in 0..3, Wdw/Wpw -1..1.  Depthwise |acc_dw| <= 9*3*1 = 27 -> dw in 0..27 (ReLU).
     * Pointwise |acc| <= C_in*27*1 = 1728 -> |acc*17/16| = 1836 < 2048, so the HMX
     * 12-bit requant field is exact (no saturation) and bit-exactness holds. */
    for (int i = 0; i < IH*IW*C_IN; i++) in[i]  = (uint8_t)(hvx_lcg(&s) % 4);           /* 0..3 */
    for (int i = 0; i < C_IN*KH*KW; i++) Wdw[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1); /* -1..1 */
    for (int i = 0; i < C_OUT*C_IN; i++) Wpw[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1); /* -1..1 */
    for (int c = 0; c < C_OUT;      c++) bias[c] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0; /* relu-zero, uint8-sat-high, passthrough */

    /* --- depthwise reference (scalar; VALID 3x3, per-channel) --- */
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < C_IN; c++) {
                int acc = 0;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        acc += (int)in[((oh*STRIDE + kh)*IW + (ow*STRIDE + kw))*C_IN + c]
                             * (int)Wdw[c*(KH*KW) + kh*KW + kw];
                dw[(oh*OW + ow)*C_IN + c] = saturate_u8(acc > 0 ? acc : 0);
            }

    /* --- pointwise reference via HVX vrmpy (K=C_in padded to 128) --- */
    for (int i = 0; i < P_OUT*KPAD; i++) act8[i] = 0;
    for (int i = 0; i < C_OUT*KPAD; i++) Wt8[i] = 0;
    for (int p = 0; p < P_OUT; p++)
        for (int c = 0; c < C_IN; c++) act8[p*KPAD + c] = dw[p*C_IN + c];
    for (int co = 0; co < C_OUT; co++)
        for (int c = 0; c < C_IN; c++) Wt8[co*KPAD + c] = Wpw[co*C_IN + c];

    const HVX_Vector zero = Q6_V_vzero();
    for (int p = 0; p < P_OUT; p++) {
        HVX_Vector va = *(const HVX_Vector *)(act8 + p*KPAD);
        for (int co = 0; co < C_OUT; co++) {
            HVX_Vector vb = *(const HVX_Vector *)(Wt8 + co*KPAD);
            HVX_Vector vp = Q6_Vw_vrmpyacc_VwVbVb(zero, va, vb);
            int acc = (int)hreduce32(vp);
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            ref[p*C_OUT + co] = saturate_u8(relu);
        }
    }

    /* Independent scalar golden on output position p=0 — guards the vrmpy reference. */
    {
        int p = 0;
        for (int co = 0; co < C_OUT; co++) {
            int acc = 0;
            for (int c = 0; c < C_IN; c++) acc += (int)dw[p*C_IN + c] * (int)Wpw[co*C_IN + c];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            uint8_t g = saturate_u8(relu);
            if (g != ref[p*C_OUT + co]) {
                printf("HVXENV_REFCHECK_FAIL co=%d scalar=%d vrmpy=%d\n",
                       co, (int)g, (int)ref[p*C_OUT + co]);
                return 2;
            }
        }
    }

    for (int i = 0; i < P_OUT*C_OUT; i++) *((volatile unsigned char *)&out[i]) = 0xA5; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, Wdw, Wpw, bias, out, P_OUT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < P_OUT*C_OUT; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, P_OUT*C_OUT, fb, gotv, expv);
    return errors ? 1 : 0;
}
