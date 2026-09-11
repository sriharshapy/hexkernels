/* HMX int8 linear projection + bias + saturating requant-to-int8 (bit-exact)
 * -- the ACCELERATED expert. Identical HMX tiling/crouton flow as
 * i8_matmul_hmx_64x64 (2x2 grid of 32x32 output tiles, each accumulating over
 * 2 Din-tiles), but the weight is packed straight from W's natural
 * [Dout,Din] nn.Linear row-major layout with swapped tile indices
 * (weight[k][o] = W^T[k][o] = W[o][k] = W[(tj*T+o)*Din + (kt*T+k)]) -- same
 * trick as the attention QK^T sibling task, since a Linear weight's natural
 * layout is already "pre-transposed" relative to a generic A.B matmul's B
 * operand. The final unpack loop FUSES bias-add -> saturating narrow into
 * the HVX epilogue (no ReLU -- this is a plain projection, not an
 * activation-fused GEMM). Crouton pack/unpack is staged through cacheable
 * buffers and moved to/from VTCM in bulk 128B vector copies (scalar VTCM
 * access costs ~48 cyc/access in timing mode and would lose to the baseline).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *X, const int8_t *W, const int32_t *bias,
                       int8_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u); /* 2048B */
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u); /* 1024B */
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;   /* cacheable crouton staging */
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;    /* even/low bytes stay 0 */
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40; /* HMX requant config: scale 17/16, bias 0 */

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32;            /* crouton tile edge */
    const int nt = n / T;        /* tiles per axis (2 for n=64) */

    for (int ti = 0; ti < nt; ti++) {          /* token-row tile */
        for (int tj = 0; tj < nt; tj++) {      /* output-channel tile */
            __asm__ volatile("mxclracc\n");                 /* clear accumulator per out tile */
            for (int kt = 0; kt < nt; kt++) {  /* Din-axis (reduction) tile */
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = X[(ti*T + i)*n + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int o = 0; o < T; o++)
                        /* weight[k][o] = W^T[k][o] = W[o][k]: read W directly with
                         * swapped tile indices -- no transpose buffer needed */
                        aWgt[hvx_hmx_i8_wgt_off(k, o)] = W[(tj*T + o)*n + (kt*T + k)];
                { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
                  for (int b = 0; b < 16; b++) d[b] = s[b]; }
                { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
                  for (int b = 0; b < 8; b++) d[b] = s[b]; }
                __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                                 :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
            }
            __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
            __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
            __asm__ volatile("isync\n\t");
            { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
              for (int b = 0; b < 16; b++) d[b] = s[b]; }
            /* Fused HVX epilogue, vectorised.  The crouton output layout is
             * off(r,c) = (r/2)*64 + c*2 + (r&1), so ONE 128B vector is exactly one
             * row PAIR: 32 columns of the even row interleaved with 32 of the odd.
             * vshuffe/vshuffo split it into the two rows -- both land in the LOW 32
             * lanes, which is why only Q6_V_lo_W of the widening pair is taken (a
             * vdeal + lo/hi version failed exactly half the elements, the signature
             * of a wrong upper-half ordering assumption).  vasl/vasr sign-extends the
             * 12-bit requant field, vsxt widens to int32, the per-output-channel bias
             * is a word-lane vadd, and the narrow to int8 is the vpack:sat chain
             * (word -> halfword -> byte), so the clamp to [-128,127] is the hardware's
             * own -- exact here because saturation is monotone, so saturating twice
             * equals clamping once.  Each vpack takes the SAME vector as both operands
             * so the low 32 bytes hold this row's values whichever operand fills the
             * low half.  bias[] is HVX_ALIGN and tj*T is a multiple of 32 words, so
             * the bias load is aligned; the final hand-off to out[] is scalar because
             * a 32-column int8 tile row is 32 B while out's row stride is 64 B, so no
             * aligned vector store covers it. */
            {
                const HVX_Vector vb = *(const HVX_Vector *)(bias + tj*T);
                static int8_t aPack[128] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector v  = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(v, v), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(v, v), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb);  /* row 2*rp   */
                    w[1] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb);  /* row 2*rp+1 */
                    for (int half = 0; half < 2; half++) {
                        HVX_Vector h = Q6_Vh_vpack_VwVw_sat(w[half], w[half]);
                        *(HVX_Vector *)aPack = Q6_Vb_vpack_VhVh_sat(h, h);
                        int8_t *dst = out + (ti*T + 2*rp + half)*n + tj*T;
                        for (int o = 0; o < T; o++) dst[o] = aPack[o];
                    }
                }
            }
        }
    }
}
