/* sol_04: HVX fp32 ReLU6 — 2-vector unrolled for throughput. */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

void candidate_kernel(const float *x, float *out, int n) {
    int vec_size = 32;
    int i = 0;

    float zero_f = 0.0f, six_f = 6.0f;
    unsigned int zero_bits, six_bits;
    __builtin_memcpy(&zero_bits, &zero_f, 4);
    __builtin_memcpy(&six_bits,  &six_f,  4);
    HVX_Vector vzero = Q6_V_vsplat_R(zero_bits);
    HVX_Vector vsix  = Q6_V_vsplat_R(six_bits);

    for (; i <= n - 2 * vec_size; i += 2 * vec_size) {
        HVX_Vector vx0 = *(const HVX_Vector *)(x + i);
        HVX_Vector vx1 = *(const HVX_Vector *)(x + i + vec_size);
        *(HVX_Vector *)(out + i)            = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(vx0, vzero), vsix);
        *(HVX_Vector *)(out + i + vec_size) = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(vx1, vzero), vsix);
    }
    for (; i <= n - vec_size; i += vec_size) {
        HVX_Vector vx = *(const HVX_Vector *)(x + i);
        *(HVX_Vector *)(out + i) = Q6_Vsf_vmin_VsfVsf(Q6_Vsf_vmax_VsfVsf(vx, vzero), vsix);
    }
    for (; i < n; i++) {
        float v = x[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        out[i] = v;
    }
}
