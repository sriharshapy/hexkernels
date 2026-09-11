/* EXPERT (achievability bar) — the L3 composition: HMX x2 + VTCM + HVX epilogue.
 * A transformer FFN / MLP block: H = ReLU(X*W1 + b1), out = H*W2 + b2. BOTH
 * matmuls run on the HMX matrix engine (32x32 crouton tiles); the intermediate
 * activation H is staged VTCM-resident between the two matmuls; the requant/ReLU/
 * bias/saturate epilogues run on HVX/scalar over cacheable crouton copies.
 *
 * WHY THIS BEATS THE HVX vrmpy BASELINE: the baseline computes both matmuls as
 * S*Dff*D + S*D*Dff vrmpy dot-products with a horizontal reduce per output; the
 * expert offloads all of that arithmetic to the HMX systolic array (one packet per
 * 32x32x32 tile-mul, accumulating across K-tiles), spending HVX only on packing +
 * the pointwise epilogues.
 *
 * CROUTON PACK/UNPACK IS THE REAL COST — it is staged through cacheable buffers and
 * moved to/from VTCM in bulk 128B vector copies (a scalar VTCM access costs ~48 cyc
 * in timing mode and would lose). H lives in VTCM between the matmuls (bulk-copied
 * in, bulk-copied back out for the second matmul's activation packing).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md +
 *         datasets/v5/tasks/i8_matmul_bias_relu_hmx/expert.c */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

#define ACT_BYTES 2048u
#define WGT_BYTES 1024u

void candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      int8_t *out, int S, int D, int Dff) {
    const uint32_t vAct  = HVX_VTCM_BASE + 0x0000u;   /* 2KB activation crouton   */
    const uint32_t vWgt  = HVX_VTCM_BASE + 0x0800u;   /* 1KB weight crouton       */
    const uint32_t vBias = HVX_VTCM_BASE + 0x1000u;   /* 2KB requant config       */
    const uint32_t vOut  = HVX_VTCM_BASE + 0x1800u;   /* 2KB output crouton       */
    const uint32_t vH    = HVX_VTCM_BASE + 0x2000u;   /* S*Dff intermediate H     */

    static uint8_t  aAct[ACT_BYTES]  HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[WGT_BYTES]  HVX_ALIGN;
    static uint16_t aOut[32*32]      HVX_ALIGN;
    static uint8_t  aBias[ACT_BYTES] HVX_ALIGN;
    static uint8_t  aH [256*256]     HVX_ALIGN;   /* H produced by matmul1     */
    static uint8_t  aH2[256*256]     HVX_ALIGN;   /* H read back from VTCM     */

    for (int i = 0; i < (int)ACT_BYTES; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < (int)ACT_BYTES; i++) aBias[i] = 0x40; /* requant: scale 17/16  */
    { HVX_Vector *sb = (HVX_Vector *)aBias, *db = (HVX_Vector *)(uintptr_t)vBias;
      for (int b = 0; b < 16; b++) db[b] = sb[b]; }           /* config -> VTCM once   */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;
    const int ntS = S / T, ntDff = Dff / T, ntD = D / T;

    /* ================= matmul1: acc1 = X . W1 -> ReLU+requant -> H ================= */
    for (int ti = 0; ti < ntS; ti++) {
        for (int tj = 0; tj < ntDff; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < ntD; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = X[(ti*T + i)*D + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W1[(kt*T + k)*Dff + (tj*T + j)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)(uintptr_t)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)(uintptr_t)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            /* Fused HVX epilogue 1: bias -> ReLU -> requant. One 128B vector is exactly
             * one crouton row PAIR (off(r,c) = (r/2)*64 + c*2 + (r&1)); vshuffe/vshuffo
             * put both rows in the LOW 32 lanes, which is why only Q6_V_lo_W of the
             * widening pair is taken, and the vasl/vasr pair sign-extends the 12-bit
             * requant field. ffn_relu_requant is clamp(max(p1,0) >> FFN_SH1, 0, 127):
             * the max against zero makes the "p1 > 0" test unnecessary, since a
             * non-positive p1 becomes 0 and 0 >> FFN_SH1 is still 0. The hand-off to
             * aH[] is scalar because a 32-column uint8 tile row is 32 B. */
            {
                const HVX_Vector vzero = Q6_V_vzero();
                const HVX_Vector v127  = Q6_V_vsplat_R(127);
                const HVX_Vector vb    = *(const HVX_Vector *)(b1 + tj*T);
                static int32_t aRow[32] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb);
                    w[1] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb);
                    for (int h = 0; h < 2; h++) {
                        *(HVX_Vector *)aRow = Q6_Vw_vmin_VwVw(
                            Q6_Vw_vasr_VwR(Q6_Vw_vmax_VwVw(w[h], vzero), FFN_SH1), v127);
                        uint8_t *dst = aH + (ti*T + 2*rp + h)*Dff + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = (uint8_t)aRow[j];
                    }
                }
            }
        }
    }

    /* stage H VTCM-resident (bulk copy), then read it back for matmul2 packing */
    const int nvH = (S*Dff) / 128;
    { HVX_Vector *s = (HVX_Vector *)aH, *d = (HVX_Vector *)(uintptr_t)vH;
      for (int b = 0; b < nvH; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vH, *d = (HVX_Vector *)aH2;
      for (int b = 0; b < nvH; b++) d[b] = s[b]; }

    /* ================= matmul2: acc2 = H . W2 -> requant+bias+sat -> out ============ */
    for (int ti = 0; ti < ntS; ti++) {
        for (int tj = 0; tj < ntD; tj++) {
            __asm__ volatile("mxclracc\n");
            for (int kt = 0; kt < ntDff; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = aH2[(ti*T + i)*Dff + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = W2[(kt*T + k)*D + (tj*T + j)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)(uintptr_t)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)(uintptr_t)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)(uintptr_t)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            /* Fused HVX epilogue 2: bias -> requant shift -> saturate to int8, same
             * crouton row-pair unpack as epilogue 1. Q6_Vw_vasr_VwR is the arithmetic
             * shift the reference's `>>` performs on a signed int, so negatives round
             * the same way. The clamp to [-128,127] is left in word lanes (vmax/vmin)
             * rather than a vpack:sat chain, because the 32-column int8 tile row is
             * 32 B against a D-byte row stride and is handed off scalar anyway. */
            {
                const HVX_Vector vlo = Q6_V_vsplat_R(-128);
                const HVX_Vector vhi = Q6_V_vsplat_R(127);
                const HVX_Vector vb  = *(const HVX_Vector *)(b2 + tj*T);
                static int32_t aRow[32] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb), FFN_SH2);
                    w[1] = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb), FFN_SH2);
                    for (int h = 0; h < 2; h++) {
                        *(HVX_Vector *)aRow =
                            Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w[h], vlo), vhi);
                        int8_t *dst = out + (ti*T + 2*rp + h)*D + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = (int8_t)aRow[j];
                    }
                }
            }
        }
    }
}
