/* Near-miss: identical HMX deep-K flow, but packs the weight WITHOUT the
 * required index swap -- i.e. treats K as if it were already the generic
 * B[D,S] matmul operand (weight[r][c] = K[(kt*T+r)*n+c]) instead of correctly
 * reading K's [S,D] row-major layout as K^T (weight[r][c] = K[c*k_dim+(kt*T+r)]).
 * This computes Q.K (not Q.K^T) -- compiles, uses real HMX, but is
 * numerically wrong -> must FAIL the tolerance gate. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *out, int n, int k_dim) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    const int T = 32;
    const int kt_count = k_dim / T;

    Q6_mxclracc_hf();
    for (int kt = 0; kt < kt_count; kt++) {
        for (int r = 0; r < n; r++)
            for (int c = 0; c < T; c++)
                aA[hvx_crouton_off(r, c)] = Q[r*k_dim + kt*T + c];
        for (int r = 0; r < T; r++)
            for (int c = 0; c < n; c++)
                /* BUG: no index swap, and reads K as if [D,S] shaped -- but K
                 * is only n*k_dim=32*128 elements, so this reads K with the
                 * wrong stride (n instead of k_dim), computing something
                 * numerically unrelated to Q.K^T */
                aB[hvx_crouton_off(r, c)] = K[(kt*T + r)*n + c];
        { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
        Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    }
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out[r*n + c] = aO[hvx_crouton_off(r, c)];
}
