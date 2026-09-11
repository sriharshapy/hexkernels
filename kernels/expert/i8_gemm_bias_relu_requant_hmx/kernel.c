/* HMX int8 64x64 GEMM + bias + ReLU + saturating requant-to-uint8 (bit-exact)
 * — the ACCELERATED expert. Identical HMX tiling/crouton flow as
 * i8_matmul_hmx_64x64 (2x2 grid of 32x32 output tiles, each accumulating over
 * 2 K-tiles), but the final unpack loop FUSES bias-add -> ReLU -> saturating
 * narrow into the HVX epilogue -- the classic quantized-inference matmul
 * epilogue chain. Crouton pack/unpack is staged through cacheable buffers and
 * moved to/from VTCM in bulk 128B vector copies (scalar VTCM access costs
 * ~48 cyc/access in timing mode and would lose to the baseline).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

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

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       uint8_t *out, int n) {
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

    for (int ti = 0; ti < nt; ti++) {
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");                 /* clear accumulator per out tile */
            for (int kt = 0; kt < nt; kt++) {
                for (int i = 0; i < T; i++)
                    for (int k = 0; k < T; k++)
                        aAct[hvx_hmx_i8_act_off(i, k)] = A[(ti*T + i)*n + (kt*T + k)];
                for (int k = 0; k < T; k++)
                    for (int j = 0; j < T; j++)
                        aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[(kt*T + k)*n + (tj*T + j)];
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
            /* Fused HVX epilogue: sign-extend + bias add + ReLU + saturate-to-uint8.
             * The crouton output layout is off(r,c) = (r/2)*64 + c*2 + (r&1), so ONE
             * 128B vector is exactly one row PAIR: 32 columns of the even row
             * interleaved with 32 of the odd. vshuffe/vshuffo split it into the two
             * rows -- both land in the LOW 32 lanes, which is why only Q6_V_lo_W of
             * the widening pair is taken (a vdeal + lo/hi version failed exactly half
             * the elements, the signature of a wrong upper-half ordering assumption).
             * The vasl/vasr pair sign-extends the 12-bit requant field, vsxt widens to
             * int32, and then the whole epilogue runs in vector lanes: vadd for the
             * per-column bias (bias[] is HVX_ALIGN and tj*T is a multiple of 32 words,
             * so the load is aligned), vmax against zero for the ReLU, and the vpack:sat
             * chain word -> halfword -> UNSIGNED byte for the requant, which puts the
             * [0,255] clamp in hardware -- exact because saturation is monotone, so
             * saturating twice equals clamping once. Each vpack takes the same vector as
             * both operands so the low 32 bytes hold this row whichever operand fills the
             * low half. The hand-off to out[] is scalar: a 32-column uint8 tile row is
             * 32 B against an n-byte row stride, so no aligned vector store covers it. */
            {
                const HVX_Vector vzero = Q6_V_vzero();
                const HVX_Vector vb    = *(const HVX_Vector *)(bias + tj*T);
                static uint8_t aPack[128] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb), vzero);
                    w[1] = Q6_Vw_vmax_VwVw(Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb), vzero);
                    for (int h = 0; h < 2; h++) {
                        HVX_Vector hw = Q6_Vh_vpack_VwVw_sat(w[h], w[h]);
                        *(HVX_Vector *)aPack = Q6_Vub_vpack_VhVh_sat(hw, hw);
                        uint8_t *dst = out + (ti*T + 2*rp + h)*n + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = aPack[j];
                    }
                }
            }
        }
    }
}
