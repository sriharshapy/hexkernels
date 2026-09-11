/* Solution 1 (expert path): general tiled HMX matmul. Iterates a grid of 32x32
 * output tiles (1x1 here), each accumulating over K/32 K-tiles: mxclracc, one
 * matmul packet per K-tile, one 0x40-config requant store, then unpack the
 * sign-extended 12-bit field to int32. Crouton pack/unpack staged via cacheable
 * buffers + bulk 128B vector copies. (K == n.) */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32, nt = n / T;

    for (int ti = 0; ti < nt; ti++)
        for (int tj = 0; tj < nt; tj++) {
            __asm__ volatile("mxclracc\n");
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
            /* Fused HVX epilogue. The crouton output layout is
             * off(r,c) = (r/2)*64 + c*2 + (r&1), so ONE 128B vector is exactly one row PAIR:
             * 32 columns of the even row interleaved with 32 of the odd. vshuffe/vshuffo split
             * it into the two rows -- both land in the LOW 32 lanes, which is why only
             * Q6_V_lo_W of the widening pair is taken (a vdeal + lo/hi version failed exactly
             * half the elements, the signature of a wrong upper-half ordering assumption).
             * The vasl/vasr pair sign-extends the 12-bit requant field, vsxt widens to int32,
             * and the store happens in word lanes. */
            {
                for (int rp = 0; rp < T/2; rp++) {
                    HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
                    HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
                    HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
                    HVX_Vector r0 = Q6_V_lo_W(Q6_Ww_vsxt_Vh(se));   /* row 2*rp   */
                    HVX_Vector r1 = Q6_V_lo_W(Q6_Ww_vsxt_Vh(so));   /* row 2*rp+1 */
                    *(HVX_Vector *)(out + (ti*T + 2*rp)*n + tj*T) = r0;
                    *(HVX_Vector *)(out + (ti*T + 2*rp + 1)*n + tj*T) = r1;
                }
            }
        }
}
