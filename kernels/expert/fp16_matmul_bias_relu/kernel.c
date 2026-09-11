/* HVX fp16 32x32x128 (deep-K) matmul + per-column bias add + ReLU (tolerance-
 * correct) -- the ACCELERATED expert (pure HVX qf16, no matrix engine).
 * Row-broadcast/AXPY qf16 matmul over the full K=128 reduction. The bias+ReLU
 * epilogue is FUSED as native HVX fp16 vector ops (Q6_Vqf16_vadd_VhfVhf +
 * Q6_Vhf_vmax_VhfVhf) directly on the matmul result vector -- NOT a scalar
 * (float) cast per element, which would round-trip through the software
 * hf<->float conversion routine 2x/element and swamp the matmul cost
 * entirely (measured ~60K extra cycles). This beats a plain scalar fp16
 * implementation by amortizing the K-reduction and the epilogue over 32
 * lanes/vector instead of one element at a time. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, const hvx_hf *bias,
                      hvx_hf *out, int n, int k_dim) {
    static hvx_hf biasPad[64] HVX_ALIGN;   /* bias[0..n-1] then zero-pad to 64 lanes */
    for (int j = 0; j < n; j++) biasPad[j] = bias[j];
    for (int j = n; j < 64; j++) biasPad[j] = (hvx_hf)0.0f;
    HVX_Vector biasVec = *(const HVX_UVector *)biasPad;
    HVX_Vector zeroVec = Q6_Vh_vsplat_R(0);

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
        HVX_Vector res = Q6_Vhf_equals_Vqf16(acc);         /* qf16 -> fp16 */
        /* v68 lacks a native Vhf+Vhf add instruction (that needs v79) -- do the
         * bias add in qf16 (native at v68, same path as the matmul accumulate)
         * then convert once, and ReLU with the native Vhf max (v68-native). */
        HVX_Vector biasedQf = Q6_Vqf16_vadd_VhfVhf(res, biasVec);
        HVX_Vector biased = Q6_Vhf_equals_Vqf16(biasedQf);
        HVX_Vector relued = Q6_Vhf_vmax_VhfVhf(biased, zeroVec);
        const hvx_hf *rp = (const hvx_hf *)&relued;
        for (int j = 0; j < n; j++) out[i*n + j] = rp[j];  /* plain hf copy, no conversion */
    }
}
