/* Near-miss: identical HVX qf16 deep-K matmul + fused bias/ReLU epilogue flow
 * as expert.c, but the K-reduction loop drops the LAST K term (off-by-one
 * loop bound: `k < k_dim - 1` instead of `k < k_dim`). Compiles, genuinely
 * uses HVX, but is numerically wrong (a full K=128-term dot product changes
 * substantially when one term of ~equal magnitude is silently dropped) ->
 * must FAIL the tolerance gate. Guards against credit for merely "uses HVX
 * qf16 + the right epilogue ops" without the correct full reduction. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim) {
    static hvx_hf biasPad[64] HVX_ALIGN;
    for (int j = 0; j < n; j++) biasPad[j] = bias[j];
    for (int j = n; j < 64; j++) biasPad[j] = (hvx_hf)0.0f;
    HVX_Vector biasVec = *(const HVX_UVector *)biasPad;
    HVX_Vector zeroVec = Q6_Vh_vsplat_R(0);

    for (int i = 0; i < n; i++) {
        unsigned short a0 = *(const unsigned short *)&A[i*k_dim + 0];
        HVX_Vector sa = Q6_Vh_vsplat_R((int)a0);
        HVX_Vector vb = *(const HVX_UVector *)(B + 0);
        HVX_Vector acc = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
        for (int k = 1; k < k_dim - 1; k++) {   /* bug: drops the last K term */
            unsigned short ak = *(const unsigned short *)&A[i*k_dim + k];
            sa = Q6_Vh_vsplat_R((int)ak);
            vb = *(const HVX_UVector *)(B + k*n);
            HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
            acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, prod);
        }
        HVX_Vector res = Q6_Vhf_equals_Vqf16(acc);
        HVX_Vector biasedQf = Q6_Vqf16_vadd_VhfVhf(res, biasVec);
        HVX_Vector biased = Q6_Vhf_equals_Vqf16(biasedQf);
        HVX_Vector relued = Q6_Vhf_vmax_VhfVhf(biased, zeroVec);
        const hvx_hf *rp = (const hvx_hf *)&relued;
        for (int j = 0; j < n; j++) out[i*n + j] = rp[j];
    }
}
