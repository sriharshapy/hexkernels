/* HMX fp16 32x32x128 (deep-K) matmul + leaky ReLU (tolerance-correct) -- the
 * ACCELERATED expert. Identical deep-K HMX flow as fp16_matmul_hmx_deepk;
 * leaky ReLU fused as native HVX vector ops on the crouton-order output
 * block: pos=max(v,0), neg=min(v,0) (both native at v68), neg*0.125 (native
 * qf16 mpy), pos+scaled (qf16 add, since v68 lacks a native Vhf+Vhf add),
 * convert once, then the plain (no-conversion) unpack. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    const int T = 32;
    const int kt_count = k_dim / T;   /* 4 for k_dim=128 */

    Q6_mxclracc_hf();
    for (int kt = 0; kt < kt_count; kt++) {
        for (int r = 0; r < n; r++)
            for (int c = 0; c < T; c++)
                aA[hvx_crouton_off(r, c)] = A[r*k_dim + kt*T + c];
        for (int r = 0; r < T; r++)
            for (int c = 0; c < n; c++)
                aB[hvx_crouton_off(r, c)] = B[(kt*T + r)*n + c];
        { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
        Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    }
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    hvx_hf slopeConst = (hvx_hf)0.125f;
    HVX_Vector slopeVec = Q6_Vh_vsplat_R((int)(*(unsigned short *)&slopeConst));
    HVX_Vector zeroVec = Q6_Vh_vsplat_R(0);

    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) {
          HVX_Vector v = s[b];
          HVX_Vector pos = Q6_Vhf_vmax_VhfVhf(v, zeroVec);
          HVX_Vector neg = Q6_Vhf_vmin_VhfVhf(v, zeroVec);
          HVX_Vector negScaledQf = Q6_Vqf16_vmpy_VhfVhf(neg, slopeVec);
          HVX_Vector sumQf = Q6_Vqf16_vadd_Vqf16Vhf(negScaledQf, pos);
          d[b] = Q6_Vhf_equals_Vqf16(sumQf);
      } }
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++)
            out[r*n + c] = aO[hvx_crouton_off(r, c)];   /* plain hf copy, no conversion */
}
