/* sol_03: HVX-vectorized sigmoid backward using Q6_Vqf32_* intrinsics.
 * dy * y * (1-y) = dy * (y - y^2); both forms are equivalent.
 * Uses qf32 fused ops: vmpy then vsub then vmpy.
 * 32 fp32 lanes per 128-byte vector.
 * Semantics: dx[i] = dy[i] * y[i] * (1.0f - y[i]) */
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const float *y, const float *dy, float *dx, int n) {
    int nvec = n / 32;
    int rem  = n - nvec * 32;

    for (int k = 0; k < nvec; k++) {
        HVX_Vector vy  = *(const HVX_Vector *)(y  + k * 32);
        HVX_Vector vdy = *(const HVX_Vector *)(dy + k * 32);

        /* Convert sf inputs to qf32 */
        HVX_Vector vy_q  = Q6_Vqf32_vadd_VsfVsf(vy,  Q6_V_vzero());
        HVX_Vector vdy_q = Q6_Vqf32_vadd_VsfVsf(vdy, Q6_V_vzero());

        /* ones vector: 1.0f broadcast */
        /* Build 1.0f constant vector: vsplat with float bit pattern */
        HVX_Vector vones_sf = Q6_V_vsplat_R(0x3F800000u); /* 1.0f */
        HVX_Vector vones_q  = Q6_Vqf32_vadd_VsfVsf(vones_sf, Q6_V_vzero());

        /* 1 - y */
        HVX_Vector v1my_q = Q6_Vqf32_vsub_Vqf32Vqf32(vones_q, vy_q);

        /* y * (1-y) */
        HVX_Vector vvar_q = Q6_Vqf32_vmpy_Vqf32Vqf32(vy_q, v1my_q);

        /* dy * [y*(1-y)] */
        HVX_Vector vdx_q  = Q6_Vqf32_vmpy_Vqf32Vqf32(vdy_q, vvar_q);

        /* Convert back to sf */
        HVX_Vector vdx_sf = Q6_Vsf_equals_Vqf32(vdx_q);
        *(HVX_Vector *)(dx + k * 32) = vdx_sf;
    }

    /* Scalar tail */
    int base = nvec * 32;
    for (int i = 0; i < rem; i++)
        dx[base + i] = dy[base + i] * y[base + i] * (1.0f - y[base + i]);
}
