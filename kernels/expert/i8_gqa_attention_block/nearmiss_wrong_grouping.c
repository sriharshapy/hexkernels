/* NEAR-MISS (must score INCORRECT): wrong GQA head-grouping. Identical HMX
 * QK^T -> softmax -> A.V flow with shared-KV-head weight-crouton reuse, but
 * forgets to advance the QUERY pointer per head (a plausible copy-paste bug when
 * generalizing a single-head attention block to GQA) -- every query head reads
 * Q's HEAD-0 data. Compiles, uses real HMX, correct for h_q==0, but wrong for
 * h_q>=1 (head1's whole output range) -> must FAIL the gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define T  32
#define NT 2

#define VT_ACT   (HVX_VTCM_BASE + 0x0000u)
#define VT_KWGT  (HVX_VTCM_BASE + 0x1000u)
#define VT_VWGT  (HVX_VTCM_BASE + 0x2000u)
#define VT_BIAS  (HVX_VTCM_BASE + 0x3000u)
#define VT_OUT   (HVX_VTCM_BASE + 0x4000u)

static inline int sx12(int field) { int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }
static inline int clamp_i8(int v) { if (v > 127) return 127; if (v < -128) return -128; return v; }

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

static void hmx_mm_resident(const uint8_t *A, int8_t *vTiles[NT][NT], uint16_t *C,
                            int Mt, int Nt, int Kt, int Kd, int ldC) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)VT_ACT;
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)VT_BIAS;
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)VT_OUT;
    static uint8_t  aAct[2048] HVX_ALIGN;
    static uint16_t aOut[T*T]  HVX_ALIGN;
    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    for (int i = 0; i < 2048; i++) aAct[i] = 0;
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
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    static uint16_t scores[GQA_S*GQA_S] HVX_ALIGN;
    static int8_t   scaled[GQA_S*GQA_S] HVX_ALIGN;
    static uint8_t  probs[GQA_S*GQA_S]  HVX_ALIGN;

    pack_weight(K, vKwgt, GQA_D, 1);
    pack_weight(V, vVwgt, GQA_D, 0);

    for (int hq = 0; hq < GQA_H_Q; hq++) {
        const uint8_t *Qh = Q;   /* BUG: should be Q + hq*GQA_S*GQA_D */
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_D;

        hmx_mm_resident(Qh, vKwgt, scores, GQA_S/T, GQA_S/T, GQA_D/T, GQA_D, GQA_S);

        for (int i = 0; i < GQA_S; i++)
            for (int j = 0; j < GQA_S; j++)
                scaled[i*GQA_S+j] = (int8_t)clamp_i8(sx12((int)scores[i*GQA_S+j]) >> 4);
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

        hmx_mm_resident(probs, vVwgt, Oh, GQA_S/T, GQA_D/T, GQA_S/T, GQA_S, GQA_D);
    }
}
