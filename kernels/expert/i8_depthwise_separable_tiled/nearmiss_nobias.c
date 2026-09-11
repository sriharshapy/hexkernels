/* NEAR-MISS (must score INCORRECT) — identical HVX depthwise + HMX pointwise + DMA/VTCM
 * composition, but the fused epilogue OMITS the per-output-channel bias add
 * (out = saturate_u8(max(r,0))).  With bias spanning +-2000 (incl. forced -2000/+2000/0
 * edges) the output diverges.  NOTE: omitting ReLU alone would NOT discriminate here
 * (saturate_u8 already clamps negatives), so the discriminating bug is the missing bias.
 *
 * Below is the expert verbatim except the marked epilogue line.
 * L3 depthwise-separable block: HVX depthwise (per-channel)
 * -> HMX 1x1 pointwise (dense matmul) + DMA/VTCM double-buffer + fused HVX epilogue.
 *
 * COMPOSITION (three mechanism groups):
 *   - HVX depthwise: 3x3 VALID per-channel conv, NHWC, vectorized across the C_in=64
 *     channels (=64 int16 lanes = one HVX vector).  Depthwise does NOT lower to a dense
 *     matmul, so it stays on HVX -- identical to the baseline's depthwise (shared cost);
 *     only the pointwise engine differs, isolating the HMX win.
 *   - HMX pointwise: the 1x1 conv IS a dense matmul (M=P positions, K=C_in, N=C_out),
 *     run on the matrix engine in 32x32 crouton tiles.  Weight croutons are packed +
 *     uDMA'd DDR->VTCM ONCE (resident, reused across all M-tiles).  The dw activation
 *     crouton for an M-tile is packed once and DOUBLE-BUFFERED: while HMX computes M-tile
 *     ti's whole tj x kt grid from VTCM buffer (ti&1), uDMA streams M-tile ti+1's croutons
 *     into the ALTERNATE VTCM buffer -> DMA latency hides behind HMX compute.
 *   - HVX epilogue: sign-extend the HMX 12-bit requant field, add per-output-channel
 *     bias, ReLU, saturate to uint8.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md +
 *         datasets/v5/tasks/i8_gemm_tiled_bias_relu_requant/expert.c (uDMA double-buffer). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define NTM   (P_OUT / 32)       /* 8  M-tiles (output positions) */
#define NTN   (C_OUT / 32)       /* 2  N-tiles (output channels) */
#define NKT   (C_IN / 32)        /* 2  K-tiles (reduction over input channels; 64=2*32) */
#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

static desc_t dw_desc[NTN*NKT];  /* weight chain (once) */
static desc_t da_desc[2][NKT];   /* activation chain, one per double-buffer */

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

/* 3x3 VALID depthwise (per-channel), NHWC, vectorized across C_in=64 channels.
 * Writes dw16[p][c] (int16, ReLU'd, 0..27). */
static void depthwise_hvx(const uint8_t *in, const int8_t *Wdw, int16_t *dw16) {
    static int16_t wtmp[KH*KW][C_IN] HVX_ALIGN;
    for (int t = 0; t < KH*KW; t++)
        for (int c = 0; c < C_IN; c++) wtmp[t][c] = (int16_t)Wdw[c*(KH*KW) + t];
    HVX_Vector wv[KH*KW];
    for (int t = 0; t < KH*KW; t++) wv[t] = *(const HVX_Vector *)wtmp[t];

    const HVX_Vector vzero = Q6_V_vzero();
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++) {
            HVX_Vector acc = vzero;
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) {
                    const uint8_t *a = in + ((oh*STRIDE + kh)*IW + (ow*STRIDE + kw))*C_IN;
                    HVX_Vector iv = *(const HVX_UVector *)a;
                    /* Q6_Wh_vunpack_Vb widens IN ORDER (vzxt deinterleaves — do not use) */
                    HVX_Vector ivh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(iv));
                    acc = Q6_Vh_vadd_VhVh(acc, Q6_Vh_vmpyi_VhVh(ivh, wv[kh*KW + kw]));
                }
            HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, acc);
            acc = Q6_V_vmux_QVV(neg, vzero, acc);   /* ReLU */
            *(HVX_Vector *)(dw16 + (oh*OW + ow)*C_IN) = acc;
        }
}

/* pack M-tile ti's dw activation croutons (all K-tiles) into DDR staging */
static void pack_act(uint8_t stage[NKT][ACT_BYTES], const int16_t *dw16, int ti) {
    for (int kt = 0; kt < NKT; kt++)
        for (int i = 0; i < 32; i++) {
            const int16_t *row = dw16 + (ti*32 + i)*C_IN;
            for (int k = 0; k < 32; k++)
                stage[kt][hvx_hmx_i8_act_off(i, k)] = (uint8_t)row[kt*32 + k];   /* 0..27 */
        }
}

void candidate_kernel(const uint8_t *in, const int8_t *Wdw, const int8_t *Wpw,
                      const int32_t *bias, uint8_t *out, int n) {
    static int16_t dw16[P_OUT*C_IN] HVX_ALIGN;
    depthwise_hvx(in, Wdw, dw16);

    /* --- VTCM partition (all disjoint) --- */
    const uint32_t vWgtR = HVX_VTCM_BASE + 0x00000u;             /* NTN*NKT * 1KB */
    const uint32_t vActB[2] = { HVX_VTCM_BASE + 0x02000u,        /* NKT * 2KB (buf0) */
                                HVX_VTCM_BASE + 0x04000u };       /* NKT * 2KB (buf1) */
    const uint32_t vBias = HVX_VTCM_BASE + 0x06000u;            /* 2KB requant config */
    const uint32_t vOut  = HVX_VTCM_BASE + 0x07000u;            /* 2KB output crouton */

    static uint8_t  wStage[NTN*NKT][WGT_BYTES] HVX_ALIGN;
    static uint8_t  aStage[2][NKT][ACT_BYTES]  HVX_ALIGN;
    static uint16_t aOut[32*32]                HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES]           HVX_ALIGN;

    for (int t = 0; t < NTN*NKT; t++) for (int i = 0; i < (int)WGT_BYTES; i++) wStage[t][i] = 0;
    for (int b = 0; b < 2; b++) for (int t = 0; t < NKT; t++)
        for (int i = 0; i < (int)ACT_BYTES; i++) aStage[b][t][i] = 0;   /* dead low bytes */
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40;
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    /* --- pack all pointwise weights, chained uDMA DDR->VTCM once (resident) --- */
    for (int tj = 0; tj < NTN; tj++)
        for (int kt = 0; kt < NKT; kt++) {
            int8_t *st = (int8_t *)wStage[tj*NKT + kt];
            for (int k = 0; k < 32; k++)
                for (int j = 0; j < 32; j++)
                    st[hvx_hmx_i8_wgt_off(k, j)] = Wpw[(tj*32 + j)*C_IN + (kt*32 + k)];
        }
    for (int t = 0; t < NTN*NKT; t++) {
        dw_desc[t].next = (t + 1 < NTN*NKT) ? (uint32_t)(uintptr_t)&dw_desc[t+1] : 0;
        dw_desc[t].ctrl = WGT_BYTES;
        dw_desc[t].src  = (uint32_t)(uintptr_t)wStage[t];
        dw_desc[t].dst  = vWgtR + (uint32_t)t * WGT_BYTES;
    }
    Q6_dmstart_A(&dw_desc[0]); Q6_R_dmwait();

    #define ISSUE_ACT(buf) do {                                                       \
        for (int kt = 0; kt < NKT; kt++) {                                            \
            da_desc[buf][kt].next = (kt + 1 < NKT) ? (uint32_t)(uintptr_t)&da_desc[buf][kt+1] : 0; \
            da_desc[buf][kt].ctrl = ACT_BYTES;                                        \
            da_desc[buf][kt].src  = (uint32_t)(uintptr_t)aStage[buf][kt];             \
            da_desc[buf][kt].dst  = vActB[buf] + (uint32_t)kt * ACT_BYTES;            \
        }                                                                             \
        Q6_dmstart_A(&da_desc[buf][0]);                                              \
    } while (0)

    pack_act(aStage[0], dw16, 0);
    ISSUE_ACT(0); Q6_R_dmwait();

    for (int ti = 0; ti < NTM; ti++) {
        int cur = ti & 1;
        if (ti + 1 < NTM) {
            int nb = (ti + 1) & 1;
            pack_act(aStage[nb], dw16, ti + 1);
            ISSUE_ACT(nb);
        }
        for (int tj = 0; tj < NTN; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < NKT; kt++) {
                uint32_t vA  = vActB[cur] + (uint32_t)kt * ACT_BYTES;
                uint32_t vWv = vWgtR + (uint32_t)(tj*NKT + kt) * WGT_BYTES;
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vA), "r"(lim_a), "r"(vWv), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");

            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            for (int i = 0; i < 32; i++) {
                int p = ti*32 + i;
                for (int j = 0; j < 32; j++) {
                    int co     = tj*32 + j;
                    int r      = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
                    int relu   = r > 0 ? r : 0;   /* BUG: bias[co] omitted */
                    out[p*C_OUT + co] = saturate_u8(relu);
                    (void)bias;
                }
            }
        }
        if (ti + 1 < NTM) Q6_R_dmwait();
    }
    #undef ISSUE_ACT
    (void)n;
}
