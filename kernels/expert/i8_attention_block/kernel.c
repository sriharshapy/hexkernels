/* EXPERT (achievability bar) -- the marquee L3 attention BLOCK, composed:
 *
 *   [HMX] QK^T  ->  [HVX] scale+softmax  ->  [HMX] A.V
 *
 * Both matmuls run on the HMX matrix engine using the proven int8 crouton
 * packing (decoded 2026-06-12): activation int8 in the HIGH byte of an
 * fp16-crouton slot (2048B tile, lim 2047), weight 4-deep packed (1024B tile,
 * lim 1023), 0x40-config requant (acc*17+8)>>4 & 0xFFF. Operand/output crouton
 * tiles are VTCM-RESIDENT (HVX_VTCM_BASE); crouton pack/unpack is staged in
 * cacheable DDR and moved to/from VTCM in bulk 128B HVX vector copies (scalar
 * VTCM access costs ~48 cyc in timing mode, so bulk vector moves are what turn
 * the raw matmul win into an end-to-end win -- the same trick as the QK^T/A.V
 * sibling experts and the tiled-GEMM expert).
 *
 * The intermediate scores/probs live in cacheable DDR between the two matmul
 * stages and are streamed into the VTCM crouton tiles per stage (the on-target
 * analog would uDMA them; at S=D=64 the 4KB score matrix is far too small for
 * DMA to beat a bulk vector copy, so DMA is intentionally not used here).
 *
 * Mechanisms: HMX (both matmuls) + HVX (softmax + bulk copies) + VTCM
 * (operand/output tiles) = 3 groups -> L3.
 *
 * Correctness: the whole block is fixed-point integer, so the output is
 * BIT-EXACT to the scalar reference (the LUT makes the softmax exact -- no
 * tolerance needed). Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

/* Shared VTCM partition (the two matmul stages run sequentially, so they reuse
 * the same regions). */
#define VT_ACT  (HVX_VTCM_BASE + 0x0000u)  /* 2048B activation crouton */
#define VT_WGT  (HVX_VTCM_BASE + 0x1000u)  /* 1024B weight crouton */
#define VT_BIAS (HVX_VTCM_BASE + 0x2000u)  /* 2048B requant config */
#define VT_OUT  (HVX_VTCM_BASE + 0x3000u)  /* 2048B output crouton */

/* ---- one 32x32-tiled HMX matmul C[MxN] = A[MxK] . W[KxN] with 0x40 requant.
 * transpose_w=1 packs the weight from W stored [N,K] (weight[k][j]=W[j][k]) --
 * used by QK^T where K is stored [S,D] row-major. transpose_w=0 packs weight
 * straight from W[K,N] -- used by A.V. Activation is uint8. Output is the uint16
 * 12-bit requant field, row-major [M,N]. */
static void hmx_matmul(const uint8_t *A, const int8_t *W, uint16_t *C,
                       int M, int N, int Kd, int ldW, int transpose_w) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)VT_ACT;
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)VT_WGT;
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)VT_OUT;

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int ntM = M / T, ntN = N / T, nkt = Kd / T;

    for (int ti = 0; ti < ntM; ti++) {
        for (int tj = 0; tj < ntN; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < nkt; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*Kd + (kt*T + k)];
                if (transpose_w) {
                    for (int k = 0; k < T; k++)
                        for (int j = 0; j < T; j++)
                            /* weight[k][j] = W^T[k][j] = W[j][k], W stored [N,ldW] */
                            aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(tj*T + j)*ldW + (kt*T + k)];
                } else {
                    for (int k = 0; k < T; k++)
                        for (int j = 0; j < T; j++)
                            aWgt[hvx_hmx_i8_wgt_off(k, j)] = W[(kt*T + k)*ldW + (tj*T + j)];
                }
                { HVX_Vector *sp = (HVX_Vector *)aAct, *dp = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
                { HVX_Vector *sp = (HVX_Vector *)aWgt, *dp = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) dp[b] = sp[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *sp = (HVX_Vector *)vOut, *dp = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) dp[b] = sp[b]; }
            for (int i = 0; i < T; i++)
                for (int j = 0; j < T; j++)
                    C[(ti*T + i)*N + (tj*T + j)] = aOut[hvx_hmx_i8_out_off(i, j)];
        }
    }
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, int S, int D, const uint8_t *exp_lut) {
    static uint16_t scores[64*64] HVX_ALIGN;   /* QK^T requant field (cacheable) */
    static int8_t   scaled[64*64] HVX_ALIGN;
    static uint8_t  probs[64*64]  HVX_ALIGN;   /* softmax weights, uint8 */

    /* Stage 1 [HMX]: scores = Q . K^T  (K stored [S,D] -> transpose_w) */
    hmx_matmul(Q, K, scores, /*M*/S, /*N*/S, /*Kd*/D, /*ldW*/D, /*transpose_w*/1);

    /* Stage 2a [HVX]: scale = clamp_i8(sx12(scores) >> 4), in vector lanes.
     * `scores` is the row-major uint16 requant field, so 128 elements are two
     * halfword vectors and the result is one byte vector.  Sign-extending the
     * 12-bit field and applying the >>4 scale is a single shift pair: `vasl` by 4
     * lifts bit 11 to bit 15, then one `vasr` by 4+4 does the sign-extension and
     * the scale together.  `vpack:sat` performs the clamp to [-128,127] in
     * hardware, with the SECOND operand landing in the low 64 bytes. */
    for (int v = 0; v < (S*S)/128; v++) {
        HVX_Vector lo = ((const HVX_Vector *)scores)[2*v];
        HVX_Vector hi = ((const HVX_Vector *)scores)[2*v + 1];
        lo = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(lo, 4), 8);
        hi = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(hi, 4), 8);
        ((HVX_Vector *)scaled)[v] = Q6_Vb_vpack_VhVh_sat(hi, lo);
    }

    /* Stage 2b: row-wise softmax over the key axis.  The exp LUT has 511 entries,
     * far past what the 32-byte `vlut32` family can gather, and the normalise is an
     * integer divide by a per-row scalar that has no HVX equivalent and must stay
     * bit-exact -- so this half is necessarily scalar (see the prompt). */
    for (int i = 0; i < S; i++) {
        int m = scaled[i*S+0];
        for (int j = 1; j < S; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j < S; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j < S; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
    }

    /* Stage 3 [HMX]: out = probs . V  (V stored [S,D] as generic B -> no transpose) */
    hmx_matmul(probs, V, out, /*M*/S, /*N*/D, /*Kd*/S, /*ldW*/D, /*transpose_w*/0);
}
