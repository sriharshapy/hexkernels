/* HMX int8 64x64 matmul + residual add + saturating requant (bit-exact) —
 * the ACCELERATED expert. Identical HMX tiling/crouton flow as
 * i8_matmul_hmx_64x64 (2x2 grid of 32x32 output tiles, each accumulating over
 * 2 K-tiles), but the final unpack loop FUSES a residual add + saturating
 * narrow into the HVX epilogue. Crouton pack/unpack is staged through
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

void candidate_kernel(const uint8_t *A, const int8_t *B, const int8_t *residual,
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
            /* Fused HVX epilogue: sign-extend + residual add + saturate, all in vector
             * lanes. The crouton output layout is off(r,c) = (r/2)*64 + c*2 + (r&1), so
             * ONE 128B vector is exactly one row PAIR: 32 columns of the even row
             * interleaved with 32 of the odd. vshuffe/vshuffo split it into the two rows
             * -- both land in the LOW 32 lanes, which is why only Q6_V_lo_W of the
             * widening pair is taken (a vdeal + lo/hi version failed exactly half the
             * elements, the signature of a wrong upper-half ordering assumption).
             * vasl/vasr sign-extends the 12-bit requant field and vsxt widens to int32.
             *
             * The residual add and the [-128,127] clamp are word-lane vadd / vmax+vmin.
             * The residual itself is int8 and a tile row of it is only 32 B against a
             * 64 B row stride, so it is widened into the word-lane scratch by a short
             * scalar read rather than by a VectorPair widen -- deliberately. Byte->
             * halfword widening returns a pair whose lo/hi convention is NOT the low/
             * high 64 bytes, and two attempts to reach the second row through it were
             * wrong in different ways (3706 and 3729 of 4096 elements). The saving in
             * this narrow 32-element gather is not worth an unverified assumption in a
             * reference solution; the arithmetic that the prompt asks for is vector
             * either way. The store is scalar for the same width reason. */
            {
                const HVX_Vector vlo = Q6_V_vsplat_R(-128);
                const HVX_Vector vhi = Q6_V_vsplat_R(127);
                static int32_t aRes[32] HVX_ALIGN, aRow[32] HVX_ALIGN;
                for (int rp = 0; rp < T/2; rp++) {
                    int gi = ti*T + 2*rp;
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector crouton[2];
                    crouton[0] = Q6_V_lo_W(Q6_Ww_vsxt_Vh(se));   /* row gi   */
                    crouton[1] = Q6_V_lo_W(Q6_Ww_vsxt_Vh(so));   /* row gi+1 */
                    for (int h = 0; h < 2; h++) {
                        const int8_t *src = residual + (gi + h)*n + tj*T;
                        for (int j = 0; j < T; j++) aRes[j] = src[j];
                        HVX_Vector acc = Q6_Vw_vadd_VwVw(crouton[h],
                                                         *(const HVX_Vector *)aRes);
                        *(HVX_Vector *)aRow =
                            Q6_Vw_vmin_VwVw(Q6_Vw_vmax_VwVw(acc, vlo), vhi);
                        int8_t *dst = out + (gi + h)*n + tj*T;
                        for (int j = 0; j < T; j++) dst[j] = (int8_t)aRow[j];
                    }
                }
            }
        }
    }
}
