/* sol_04: HVX-vectorized fp16 multiply — 2-vector unrolled.
 * Uses Q6_Wqf32_vmpy_VhfVhf + Q6_Vhf_equals_Wqf32 for bit-exact fp16 multiply.
 * Processes 128 fp16 elements per iteration (2 vectors). */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const hvx_hf *a, const hvx_hf *b, hvx_hf *out, int n) {
    int vec_size = 64;
    int i = 0;
    for (; i <= n - 2 * vec_size; i += 2 * vec_size) {
        HVX_Vector va0 = *(const HVX_Vector *)(a + i);
        HVX_Vector vb0 = *(const HVX_Vector *)(b + i);
        HVX_Vector va1 = *(const HVX_Vector *)(a + i + vec_size);
        HVX_Vector vb1 = *(const HVX_Vector *)(b + i + vec_size);
        *(HVX_Vector *)(out + i)            = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(va0, vb0));
        *(HVX_Vector *)(out + i + vec_size) = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(va1, vb1));
    }
    for (; i <= n - vec_size; i += vec_size) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vhf_equals_Wqf32(Q6_Wqf32_vmpy_VhfVhf(va, vb));
    }
    for (; i < n; i++)
        out[i] = (hvx_hf)((float)a[i] * (float)b[i]);
}
