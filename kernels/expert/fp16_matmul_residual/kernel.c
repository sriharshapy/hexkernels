/* HVX fp16 32x32x32 matmul + elementwise residual add (tolerance-correct) --
 * the ACCELERATED expert (pure HVX qf16 matmul, no matrix engine). Row-
 * broadcast/AXPY qf16 matmul over the K=32 reduction, with the residual add
 * FUSED as a native HVX qf16 vector op (v68 lacks a native Vhf+Vhf add)
 * directly on the matmul result vector -- not a scalar (float) cast per
 * element, and NOT a per-row scalar repack of C. Instead C is read directly
 * via an unaligned 64-lane vector load at row offset i*n (mirrors how B's
 * row is read in the matmul loop -- lanes n..63 spill into the next row and
 * are discarded), with a zero-padded tail so the last row's read stays in
 * bounds. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *C,
                      hvx_hf *out, int n, int k_dim) {
    static hvx_hf cPad[32*32 + 64] HVX_ALIGN;   /* C plus a zero-padded tail */

    { const HVX_Vector *s = (const HVX_Vector *)C; HVX_Vector *d = (HVX_Vector *)cPad;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }        /* bulk copy, 1 vector/16 rows-worth */
    for (int j = n*n; j < n*n + 64; j++) cPad[j] = (hvx_hf)0.0f;

    for (int i = 0; i < n; i++) {
        unsigned short a0 = *(const unsigned short *)&A[i*k_dim + 0];
        HVX_Vector sa = Q6_Vh_vsplat_R((int)a0);
        HVX_Vector vb = *(const HVX_UVector *)(B + 0);
        HVX_Vector acc = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
        for (int k = 1; k < k_dim; k++) {
            unsigned short ak = *(const unsigned short *)&A[i*k_dim + k];
            sa = Q6_Vh_vsplat_R((int)ak);
            vb = *(const HVX_UVector *)(B + k*n);
            HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
            acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, prod);
        }
        HVX_Vector res = Q6_Vhf_equals_Vqf16(acc);   /* qf16 -> fp16 */

        HVX_Vector cVec = *(const HVX_UVector *)(cPad + i*n);
        HVX_Vector sumQf = Q6_Vqf16_vadd_VhfVhf(res, cVec);
        HVX_Vector sum = Q6_Vhf_equals_Vqf16(sumQf);
        const hvx_hf *rp = (const hvx_hf *)&sum;
        for (int j = 0; j < n; j++) out[i*n + j] = rp[j];  /* plain hf copy, no conversion */
    }
}
