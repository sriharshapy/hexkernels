/* HMX int8 FUSED attention QK^T-then-AV (bit-exact, no softmax) -- the
 * ACCELERATED expert. Two chained single-tile (S=D=32) HMX matmuls: stage 1
 * computes scores=Q.K^T (K packed with swapped tile indices -- it is already
 * "pre-transposed" relative to a generic A.B matmul's B operand, same trick
 * as the QK^T-only sibling task), requant+unpack into an intermediate uint8
 * score matrix Sc; stage 2 re-packs Sc as the activation and V (already in
 * the generic [K,N] layout, no index swap) as the weight, then requants again
 * into the final int32 output. The two stages reuse the SAME VTCM addresses
 * sequentially (stage 1 fully unpacks into a local buffer before stage 2
 * starts packing) -- no separate scratch region needed.
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                       int32_t *out, int S, int D) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u); /* 1024B */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;
    static uint8_t  Sc[32*32]   HVX_ALIGN;   /* intermediate score matrix, non-negative */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* requant config: scale 17/16, bias 0 */

    /* ---- Stage 1: scores = Q . K^T ---- */
    for (int i = 0; i < T; i++)
        for (int d = 0; d < T; d++)
            aAct[hvx_hmx_i8_act_off(i, d)] = Q[i*D + d];
    for (int d = 0; d < T; d++)
        for (int j = 0; j < T; j++)
            /* weight[d][j] = K^T[d][j] = K[j][d]: swapped tile indices, no transpose buffer */
            aWgt[hvx_hmx_i8_wgt_off(d, j)] = K[j*D + d];
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");
    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            int v = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
            Sc[i*T + j] = (uint8_t)v;   /* non-negative by construction (input range) */
        }

    /* ---- Stage 2: acc = Sc . V (V already in generic [K,N] layout) ---- */
    for (int i = 0; i < T; i++)
        for (int k = 0; k < T; k++)
            aAct[hvx_hmx_i8_act_off(i, k)] = Sc[i*T + k];
    for (int k = 0; k < T; k++)
        for (int j = 0; j < T; j++)
            aWgt[hvx_hmx_i8_wgt_off(k, j)] = V[k*D + j];
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");
    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    /* Fused HVX epilogue for the second stage. The crouton output layout is
     * off(r,c) = (r/2)*64 + c*2 + (r&1), so ONE 128B vector is exactly one row PAIR:
     * 32 columns of the even row interleaved with 32 of the odd. vshuffe/vshuffo
     * split it into the two rows -- both land in the LOW 32 lanes, which is why only
     * Q6_V_lo_W of the widening pair is taken (a vdeal + lo/hi version failed exactly
     * half the elements, the signature of a wrong upper-half ordering assumption).
     * The vasl/vasr pair sign-extends the 12-bit requant field and vsxt widens it to
     * int32. out[] is HVX_ALIGN and a row is D*4 = 128B, so the stores are aligned. */
    for (int rp = 0; rp < D/2; rp++) {
        HVX_Vector v  = ((const HVX_Vector *)aOut)[rp];
        HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(v, v), 4), 4);
        HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(v, v), 4), 4);
        *(HVX_Vector *)(out + (2*rp    )*D) = Q6_V_lo_W(Q6_Ww_vsxt_Vh(se));
        *(HVX_Vector *)(out + (2*rp + 1)*D) = Q6_V_lo_W(Q6_Ww_vsxt_Vh(so));
    }
    (void)S;
}
