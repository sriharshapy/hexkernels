/* EXPERT (achievability bar) — L3 tiled conv: im2col -> HMX matmul + DMA/VTCM
 * double-buffer + fused HVX epilogue.  int8 3x3 VALID conv lowered to a tiled matmul
 * (M=P output positions, K=C_in*Kh*Kw reduction, N=C_out).
 *
 * COMPOSITION (the L3 thing this proves — HMX and uDMA coexisting in VTCM):
 *   - WEIGHTS RESIDENT IN VTCM: every (tj,kt) weight crouton is im2col-independent and
 *     reused across all M-tiles, so it is packed + uDMA'd DDR->VTCM ONCE and kept
 *     resident.  (Re-packing weights per output tile is the classic tiling waste.)
 *   - ACTIVATION DOUBLE-BUFFERED across M-tiles: the im2col crouton for an M-tile ti is
 *     tj-independent, so we pack ALL of ti's K-tiles once and reuse them across the tj
 *     loop.  While HMX computes M-tile ti (its whole tj x kt matmul grid) from VTCM
 *     buffer (ti&1), uDMA streams M-tile ti+1's im2col croutons into the ALTERNATE VTCM
 *     buffer -> DMA latency hides behind HMX compute.  Disjoint VTCM regions -> no fault.
 *   - EPILOGUE on HVX: the HMX 0x40-config store yields the 12-bit requant field r; the
 *     epilogue sign-extends it, adds per-output-channel bias, ReLUs, saturates to uint8.
 *
 * Crouton pack is division-free (koff/poff tables) and staged in cacheable DDR (scalar
 * VTCM access is ~48 cyc in timing mode); the DDR->VTCM move is a CHAINED uDMA.  HMX
 * gives the win over the HVX vrmpy baseline; DMA+VTCM add the on-chip streaming that
 * makes this the canonical on-device tiled inference conv.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md +
 *         datasets/v5/tasks/i8_gemm_tiled_bias_relu_requant/expert.c (uDMA double-buffer). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

/* uDMA Type-0 (1D) descriptor, 16 bytes.  MUST be static/global (a stack descriptor
 * silently no-ops at -O2).  Chained via the next-ptr. */
typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define NTM   (P_OUT / 32)                 /* 8  M-tiles (output positions) */
#define NTN   (C_OUT / 32)                 /* 2  N-tiles (output channels) */
#define NKT   ((KK + 31) / 32)             /* 5  K-tiles (im2col reduction, zero-padded) */
#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

static desc_t dw_desc[NTN*NKT];            /* weight chain (once) */
static desc_t da_desc[2][NKT];             /* activation chain, one per double-buffer */

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

/* pack M-tile ti's im2col activation croutons (all K-tiles) into DDR staging */
static void pack_act(uint8_t stage[NKT][ACT_BYTES], const uint8_t *in,
                     const int *koff, const int *poff, int ti) {
    for (int kt = 0; kt < NKT; kt++) {
        for (int i = 0; i < 32; i++) {
            int po = poff[ti*32 + i];
            for (int k = 0; k < 32; k++) {
                int kg = kt*32 + k;
                stage[kt][hvx_hmx_i8_act_off(i, k)] = (kg < KK) ? in[koff[kg] + po] : 0;
            }
        }
    }
}

void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      uint8_t *out, int n) {
    /* --- VTCM partition (all disjoint) --- */
    const uint32_t vWgtR = HVX_VTCM_BASE + 0x00000u;              /* NTN*NKT * 1KB */
    const uint32_t vActB[2] = { HVX_VTCM_BASE + 0x03000u,        /* NKT * 2KB (buf0) */
                                HVX_VTCM_BASE + 0x06000u };       /* NKT * 2KB (buf1) */
    const uint32_t vBias = HVX_VTCM_BASE + 0x09000u;             /* 2KB requant config */
    const uint32_t vOut  = HVX_VTCM_BASE + 0x0A000u;             /* 2KB output crouton */

    /* Cacheable staging (crouton pack lands here; uDMA moves DDR->VTCM). */
    static uint8_t  wStage[NTN*NKT][WGT_BYTES] HVX_ALIGN;
    static uint8_t  aStage[2][NKT][ACT_BYTES]  HVX_ALIGN;
    static uint16_t aOut[32*32]                HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES]           HVX_ALIGN;

    for (int t = 0; t < NTN*NKT; t++) for (int i = 0; i < (int)WGT_BYTES; i++) wStage[t][i] = 0;
    for (int b = 0; b < 2; b++) for (int t = 0; t < NKT; t++)
        for (int i = 0; i < (int)ACT_BYTES; i++) aStage[b][t][i] = 0;   /* dead low bytes / K-pad */
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40;           /* requant scale 17/16 */
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }                     /* config -> VTCM once */

    /* division-free im2col addressing tables */
    int koff[KK], poff[P_OUT];
    for (int k = 0; k < KK; k++) {
        int ci = k / (KH*KW), r = k % (KH*KW), kh = r / KW, kw = r % KW;
        koff[k] = ci*IH*IW + kh*IW + kw;
    }
    for (int p = 0; p < P_OUT; p++) poff[p] = (p / OW)*STRIDE*IW + (p % OW)*STRIDE;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    /* --- pack all weights, chained uDMA DDR->VTCM once (resident) --- */
    for (int tj = 0; tj < NTN; tj++)
        for (int kt = 0; kt < NKT; kt++) {
            uint8_t *st = wStage[tj*NKT + kt];
            for (int k = 0; k < 32; k++) {
                int kg = kt*32 + k;
                for (int j = 0; j < 32; j++) {
                    int co = tj*32 + j;
                    ((int8_t*)st)[hvx_hmx_i8_wgt_off(k, j)] = (kg < KK) ? W[co*KK + kg] : 0;
                }
            }
        }
    for (int t = 0; t < NTN*NKT; t++) {
        dw_desc[t].next = (t + 1 < NTN*NKT) ? (uint32_t)(uintptr_t)&dw_desc[t+1] : 0;
        dw_desc[t].ctrl = WGT_BYTES;
        dw_desc[t].src  = (uint32_t)(uintptr_t)wStage[t];
        dw_desc[t].dst  = vWgtR + (uint32_t)t * WGT_BYTES;
    }
    Q6_dmstart_A(&dw_desc[0]); Q6_R_dmwait();

    /* helper to (re)issue an activation chain for buffer buf */
    #define ISSUE_ACT(buf) do {                                                       \
        for (int kt = 0; kt < NKT; kt++) {                                            \
            da_desc[buf][kt].next = (kt + 1 < NKT) ? (uint32_t)(uintptr_t)&da_desc[buf][kt+1] : 0; \
            da_desc[buf][kt].ctrl = ACT_BYTES;                                        \
            da_desc[buf][kt].src  = (uint32_t)(uintptr_t)aStage[buf][kt];             \
            da_desc[buf][kt].dst  = vActB[buf] + (uint32_t)kt * ACT_BYTES;            \
        }                                                                             \
        Q6_dmstart_A(&da_desc[buf][0]);                                              \
    } while (0)

    /* --- prologue: pack + DMA M-tile 0 activation into buffer 0 --- */
    pack_act(aStage[0], in, koff, poff, 0);
    ISSUE_ACT(0); Q6_R_dmwait();

    for (int ti = 0; ti < NTM; ti++) {
        int cur = ti & 1;

        /* Prefetch NEXT M-tile activation into the ALTERNATE buffer (async) while HMX
         * computes the current M-tile's whole tj x kt grid below. */
        if (ti + 1 < NTM) {
            int nb = (ti + 1) & 1;
            pack_act(aStage[nb], in, koff, poff, ti + 1);
            ISSUE_ACT(nb);
        }

        for (int tj = 0; tj < NTN; tj++) {
            __asm__ volatile("mxclracc\n");   /* clear HMX accumulator once per output tile */
            for (int kt = 0; kt < NKT; kt++) {
                uint32_t vA = vActB[cur] + (uint32_t)kt * ACT_BYTES;
                uint32_t vWv = vWgtR + (uint32_t)(tj*NKT + kt) * WGT_BYTES;
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vA), "r"(lim_a), "r"(vWv), "r"(lim_w) : "memory");
            }
            /* HMX requant store -> VTCM output crouton */
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");

            /* bulk-copy VTCM crouton -> cacheable, then fused HVX/scalar epilogue */
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            /* Fused HVX epilogue. The crouton output layout is off(r,c) = (r/2)*64 +
             * c*2 + (r&1), so ONE 128B vector is exactly one row PAIR: 32 columns of
             * the even row interleaved with 32 of the odd. vshuffe/vshuffo split it
             * into the two rows -- both land in the LOW 32 lanes, which is why only
             * Q6_V_lo_W of the widening pair is taken (a vdeal + lo/hi version failed
             * exactly half the elements, the signature of a wrong upper-half ordering
             * assumption). vasl/vasr sign-extends the 12-bit requant field, vsxt widens
             * to int32, and bias -> ReLU -> saturate-to-uint8 all run in word lanes --
             * the tile COLUMN j is the output channel, so bias[tj*32 + j] is indexed by
             * lane and loads as one aligned vector (C_OUT is a multiple of 32). The
             * [0,255] clamp is a vmin against 255 after the ReLU's vmax against zero.
             *
             * The STORE cannot be a vector store: out is [C_OUT][P_OUT], so a tile row
             * scatters down an output column with stride P_OUT. The 32 results go to an
             * aligned scratch in one vector store and are handed off from there. */
            {
                const HVX_Vector vzero = Q6_V_vzero();
                const HVX_Vector v255  = Q6_V_vsplat_R(255);
                const HVX_Vector vb    = *(const HVX_Vector *)(bias + tj*32);
                static int32_t aRow[32] HVX_ALIGN;
                for (int rp = 0; rp < 16; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb);
                    w[1] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb);
                    for (int h = 0; h < 2; h++) {
                        int p = ti*32 + 2*rp + h;
                        *(HVX_Vector *)aRow =
                            Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w[h], vzero), v255);
                        for (int j = 0; j < 32; j++)
                            out[(tj*32 + j)*P_OUT + p] = (uint8_t)aRow[j];
                    }
                }
            }
        }

        if (ti + 1 < NTM) Q6_R_dmwait();   /* finish the prefetch before reusing the buffer */
    }
    #undef ISSUE_ACT
    (void)n;
}
