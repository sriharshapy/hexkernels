/* HVX fp16 32x32x32 matmul + tanh (tolerance-correct) -- the ACCELERATED
 * expert (pure HVX qf16 matmul, no matrix engine). Row-broadcast/AXPY qf16
 * matmul over the K=32 reduction; tanh has no HVX vector transcendental, so
 * the epilogue is necessarily scalar -- but the matmul itself is fully
 * vectorized. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim) {
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
        const hvx_hf *rp = (const hvx_hf *)&res;
        for (int j = 0; j < n; j++)
            out[i*n + j] = (hvx_hf)tanhf((float)rp[j]);
    }
}
