/* Expert: single 32x32 HMX matrix-engine tile + saturating requant to int8.
 * Crouton pack/unpack staged via cacheable buffers + bulk 128B vector copies;
 * the 12-bit field is sign-extended then clamped to [-128,127] in the epilogue. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;

    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) aAct[hvx_hmx_i8_act_off(i, k)] = A[i*n + k];
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++) aWgt[hvx_hmx_i8_wgt_off(k, j)] = B[k*n + j];

    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
      for (int b = 0; b < 8; b++) d[b] = s[b]; }

    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    __asm__ volatile("mxclracc\n");
    __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                     :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
    __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
    __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    /* Fused HVX epilogue. The crouton output layout is off(r,c) = (r/2)*64 + c*2 +
     * (r&1), so ONE 128B vector is exactly one row PAIR: 32 columns of the even row
     * interleaved with 32 of the odd. vshuffe/vshuffo split it into the two rows --
     * both land in the LOW 32 lanes, which is why only Q6_V_lo_W of the widening pair
     * is taken (a vdeal + lo/hi version failed exactly half the elements, the signature
     * of a wrong upper-half ordering assumption). The vasl/vasr pair sign-extends the
     * 12-bit requant field, vsxt widens to int32, and the narrow back down to int8 is
     * the vpack:sat chain word -> halfword -> byte, which is where the [-128,127] clamp
     * happens in hardware -- exact because saturation is monotone, so saturating twice
     * equals clamping once. Each vpack takes the same vector as both operands so the low
     * 32 bytes hold this row whichever operand fills the low half. The hand-off to out[]
     * is scalar: a row is n = 32 bytes, too short for an aligned vector store. */
    {
        static int8_t aPack[128] HVX_ALIGN;
        for (int rp = 0; rp < n/2; rp++) {
            HVX_Vector fv = ((const HVX_Vector *)aOut)[rp];
            HVX_Vector se = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffe_VhVh(fv, fv), 4), 4);
            HVX_Vector so = Q6_Vh_vasr_VhR(Q6_Vh_vasl_VhR(Q6_Vh_vshuffo_VhVh(fv, fv), 4), 4);
            HVX_Vector w[2];
            w[0] = Q6_V_lo_W(Q6_Ww_vsxt_Vh(se));   /* row 2*rp   */
            w[1] = Q6_V_lo_W(Q6_Ww_vsxt_Vh(so));   /* row 2*rp+1 */
            for (int h = 0; h < 2; h++) {
                HVX_Vector hw = Q6_Vh_vpack_VwVw_sat(w[h], w[h]);
                *(HVX_Vector *)aPack = Q6_Vb_vpack_VhVh_sat(hw, hw);
                int8_t *dst = out + (2*rp + h)*n;
                for (int j = 0; j < n; j++) dst[j] = aPack[j];
            }
        }
    }
}
