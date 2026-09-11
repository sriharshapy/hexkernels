/* EXPERT (achievability bar) -- the L3 grouped-query attention BLOCK, composed:
 *
 *   [HMX] QK^T  ->  [HVX] scale+softmax  ->  [HMX] A.V     (per query head)
 *
 * Both matmuls run on the HMX matrix engine (proven int8 crouton packing, 0x40
 * requant). The GQA mechanism win: H_Q query heads share ONE KV head, so the
 * shared KV head's HMX weight croutons -- K (for QK^T) and V (for A.V) -- are
 * packed into VTCM ONCE, up front, and left RESIDENT; the GROUP_SIZE query heads
 * that share them reuse them without re-packing (only the query/probs activation
 * side is re-packed per head). This removes the expensive weight-crouton
 * pack+bulk-copy for all but the first head, the same reuse trick as the QK^T-only
 * sibling i8_gqa_qkt_hmx, now extended to the full block.
 *
 * PERF: scalar VTCM access costs ~48 cyc in timing mode, so ALL crouton
 * pack/unpack is STAGED through cacheable DDR buffers and moved to/from VTCM in
 * bulk 128B HVX vector copies. The scores/probs intermediates live in cacheable
 * DDR between the two matmul stages (at 4KB far too small for DMA to beat a bulk
 * vector copy). The whole block is fixed-point integer, so the LUT softmax makes
 * the output BIT-EXACT to the scalar reference (no tolerance).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define T  32
#define NT 2   /* GQA_S/T == GQA_D/T == 2 */

/* VTCM partition (K and V weight croutons stay resident across query heads). */
#define VT_ACT   (HVX_VTCM_BASE + 0x0000u)   /* 2048B activation crouton (re-packed per head/tile) */
#define VT_KWGT  (HVX_VTCM_BASE + 0x1000u)   /* NT*NT x 1024B K weight croutons (QK^T), resident   */
#define VT_VWGT  (HVX_VTCM_BASE + 0x2000u)   /* NT*NT x 1024B V weight croutons (A.V),  resident   */
#define VT_BIAS  (HVX_VTCM_BASE + 0x3000u)   /* 2048B requant config                               */
#define VT_OUT   (HVX_VTCM_BASE + 0x4000u)   /* 2048B output crouton                               */

static inline int sx12(int field) { int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }
static inline int clamp_i8(int v) { if (v > 127) return 127; if (v < -128) return -128; return v; }

/* Pack the shared KV head's weight croutons into resident VTCM tiles ONCE.
 * transpose_w=1: weight[k][j] = W[j][k]  (W stored [N, ldW]) -- QK^T (W=K over keys)
 * transpose_w=0: weight[k][j] = W[k][j]  (W stored [K, ldW]) -- A.V  (W=V) */
static void pack_weight(const int8_t *W, int8_t *vTiles[NT][NT], int ldW, int transpose_w) {
    static int8_t aWgt[1024] HVX_ALIGN;
    for (int tj = 0; tj < NT; tj++)
        for (int kt = 0; kt < NT; kt++) {
            if (transpose_w) {
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(tj*T + j)*ldW + (kt*T + k)];
            } else {
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(kt*T + k)*ldW + (tj*T + j)];
            }
            HVX_Vector *sp = (HVX_Vector *)aWgt, *dp = (HVX_Vector *)vTiles[tj][kt];
            for (int b = 0; b < 8; b++) dp[b] = sp[b];
        }
}

/* C[MxN] = A[M x Kd] . (resident weight croutons), 0x40 requant, row-major into
 * C (leading dim ldC). Only the activation is packed per output tile. */
static void hmx_mm_resident(const uint8_t *A, int8_t *vTiles[NT][NT], uint16_t *C,
                            int Mt, int Nt, int Kt, int Kd, int ldC) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)VT_ACT;
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)VT_OUT;
    static uint8_t  aAct[2048] HVX_ALIGN;
    static uint16_t aOut[T*T]  HVX_ALIGN;
    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;

    for (int i = 0; i < 2048; i++) aAct[i] = 0;   /* even/low bytes stay 0 */

    for (int ti = 0; ti < Mt; ti++) {
        for (int tj = 0; tj < Nt; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < Kt; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*Kd + (kt*T + k)];
                { HVX_Vector *sp = (HVX_Vector *)aAct, *dp = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vTiles[tj][kt]), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *sp = (HVX_Vector *)vOut, *dp = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    C[(ti*T + i)*ldC + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, const uint8_t *exp_lut) {
    int8_t *vKwgt[NT][NT], *vVwgt[NT][NT];
    for (int tj = 0; tj < NT; tj++)
        for (int kt = 0; kt < NT; kt++) {
            vKwgt[tj][kt] = (int8_t *)(uintptr_t)(VT_KWGT + (unsigned)(tj*NT + kt) * 0x400u);
            vVwgt[tj][kt] = (int8_t *)(uintptr_t)(VT_VWGT + (unsigned)(tj*NT + kt) * 0x400u);
        }
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;   /* requant: scale 17/16, bias 0 */

    static uint16_t scores[GQA_S*GQA_S] HVX_ALIGN;
    static int8_t   scaled[GQA_S*GQA_S] HVX_ALIGN;
    static uint8_t  probs[GQA_S*GQA_S]  HVX_ALIGN;

    /* Pack the shared KV head's weight croutons ONCE (H_KV == 1). */
    pack_weight(K, vKwgt, GQA_D, /*transpose_w*/1);   /* QK^T weight = K^T */
    pack_weight(V, vVwgt, GQA_D, /*transpose_w*/0);   /* A.V  weight = V   */

    for (int hq = 0; hq < GQA_H_Q; hq++) {
        const uint8_t *Qh = Q + (size_t)hq * GQA_S * GQA_D;
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_D;

        /* Stage 1 [HMX]: scores = Q . K^T  (reuse resident K croutons) */
        hmx_mm_resident(Qh, vKwgt, scores, GQA_S/T, GQA_S/T, GQA_D/T, GQA_D, GQA_S);

        /* Stage 2 [HVX/scalar]: scale (>>4 clamp) + row-wise softmax over key axis */
        /* [HVX] scale = clamp_i8(sx12(scores) >> 4), in vector lanes. `scores` is
         * the row-major uint16 requant field, so 128 elements are two halfword vectors and
         * the result is one byte vector. Sign-extending the 12-bit field and applying the
         * scale is a single shift pair: vasl by 4 lifts bit 11 to bit 15, then one vasr by
         * 4 + 4 does the sign-extension and the scale together. vpack:sat performs the
         * clamp to [-128,127] in hardware, with the SECOND operand landing in the low 64
         * bytes. */
        for (int v = 0; v < (GQA_S*GQA_S)/128; v++) {
            HVX_Vector lo = ((const HVX_Vector *)scores)[2*v];
            HVX_Vector hi = ((const HVX_Vector *)scores)[2*v + 1];
            lo = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(lo, 4), 4 + 4);
            hi = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(hi, 4), 4 + 4);
            ((HVX_Vector *)scaled)[v] = Q6_Vb_vpack_VhVh_sat(hi, lo);
        }
        for (int i = 0; i < GQA_S; i++) {
            int m = scaled[i*GQA_S+0];
            for (int j = 1; j < GQA_S; j++) if (scaled[i*GQA_S+j] > m) m = scaled[i*GQA_S+j];
            int Sr = 0;
            for (int j = 0; j < GQA_S; j++) {
                int diff = (int)scaled[i*GQA_S+j] - m;
                if (diff < -255) diff = -255;
                uint8_t e = exp_lut[diff + 255];
                probs[i*GQA_S+j] = e;
                Sr += (int)e;
            }
            int half = Sr / 2;
            for (int j = 0; j < GQA_S; j++)
                probs[i*GQA_S+j] = (uint8_t)(((int)probs[i*GQA_S+j] * 255 + half) / Sr);
        }

        /* Stage 3 [HMX]: out = probs . V  (reuse resident V croutons) */
        hmx_mm_resident(probs, vVwgt, Oh, GQA_S/T, GQA_D/T, GQA_S/T, GQA_S, GQA_D);
    }
}
