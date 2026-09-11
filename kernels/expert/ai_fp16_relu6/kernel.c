/* sol_03: HVX-vectorized fp16 ReLU6 using Q6_Vhf_vmax/vmin.
 * Clamps in fp16 native — bit-exact since 0.0 and 6.0 are exact fp16.
 * Each HVX vector holds 64 fp16 values (128 bytes / 2 bytes each).
 * Scalar tail for remainder. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    int vec_size = 64;
    int i = 0;

    /* Broadcast scalar fp16 constants into vectors */
    hvx_hf zero_hf = (hvx_hf)0.0f;
    hvx_hf six_hf  = (hvx_hf)6.0f;
    HVX_Vector vzero = Q6_Vh_vsplat_R(*(unsigned short *)&zero_hf | ((unsigned int)*(unsigned short *)&zero_hf << 16));
    HVX_Vector vsix  = Q6_Vh_vsplat_R(*(unsigned short *)&six_hf  | ((unsigned int)*(unsigned short *)&six_hf  << 16));

    for (; i <= n - vec_size; i += vec_size) {
        HVX_Vector vx = *(const HVX_Vector *)(x + i);
        HVX_Vector vclamped = Q6_Vhf_vmin_VhfVhf(Q6_Vhf_vmax_VhfVhf(vx, vzero), vsix);
        *(HVX_Vector *)(out + i) = vclamped;
    }
    /* scalar tail */
    for (; i < n; i++) {
        float v = (float)x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        out[i] = (hvx_hf)v;
    }
}
