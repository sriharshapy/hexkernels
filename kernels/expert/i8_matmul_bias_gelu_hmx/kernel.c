/* HMX int8 64x64 matmul + per-column int32 bias add + GELU-via-LUT (bit-exact)
 * — the ACCELERATED expert. Identical HMX tiling/crouton flow as
 * i8_matmul_hmx_64x64 (2x2 grid of 32x32 output tiles, each accumulating over
 * 2 K-tiles), but the final unpack loop FUSES bias-add -> saturate -> LUT
 * lookup into the HVX epilogue. Crouton pack/unpack is staged through
 * cacheable buffers and moved to/from VTCM in bulk 128B vector copies (scalar
 * VTCM access costs ~48 cyc/access in timing mode and would lose to the
 * baseline).
 * Recipe: docs/superpowers/specs/2026-06-12-hmx-int8-layout-re-agenda.md */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline int8_t saturate_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias,
                       const int8_t *gelu_lut, int8_t *out, int n) {
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
            /* fused HVX epilogue: sign-extend + bias add + saturate + LUT */
            /* Fused HVX epilogue: sign-extend -> bias -> saturate, then the GELU table
             * lookup. One 128B vector is exactly one crouton row PAIR (off(r,c) =
             * (r/2)*64 + c*2 + (r&1)); vshuffe/vshuffo put both rows in the LOW 32
             * lanes, which is why only Q6_V_lo_W of the widening pair is taken, and the
             * vasl/vasr pair sign-extends the 12-bit requant field. Bias and the
             * [-128,127] clamp then run in word lanes, and the +128 that turns the
             * saturated value into a table index is one more vadd.
             *
             * The LOOKUP stays scalar. gelu_lut has 256 entries and the HVX vlut32
             * family gathers from a 32-byte table, so a 256-entry gather costs eight
             * chained vlut32or passes -- more work than the 32 scalar loads it would
             * replace for one tile row, and this task is about the HMX matmul, not
             * about a table-gather idiom (i8_gelu_approx_lut_dma is the task that
             * exercises that). */
            {
                const HVX_Vector vlo   = Q6_V_vsplat_R(-128);
                const HVX_Vector vhi   = Q6_V_vsplat_R(127);
                const HVX_Vector v128  = Q6_V_vsplat_R(128);
                const HVX_Vector vb    = *(const HVX_Vector *)(bias + tj*T);
                static int32_t aIdx[32] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector w[2];
                    w[0] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(se)), vb);
                    w[1] = Q6_Vw_vadd_VwVw(Q6_V_lo_W(Q6_Ww_vsxt_Vh(so)), vb);
                    for (int h = 0; h < 2; h++) {
                        *(HVX_Vector *)aIdx = Q6_Vw_vadd_VwVw(
                            Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(w[h], vlo), vhi), v128);
                        int8_t *dst = out + (ti*T + 2*rp + h)*n + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = gelu_lut[(uint8_t)aIdx[j]];
                    }
                }
            }
        }
    }
}
