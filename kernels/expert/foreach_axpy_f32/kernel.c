/* sol_03: HVX-vectorized fp32 AXPY.
 * y[i] = alpha * x[i] + y[i]  via qf32 fused multiply-add.
 * Q6_Vqf32_vmpy_VsfVsf gives alpha*x in qf32, then add y via
 * Q6_Vqf32_vadd_Vqf32Vsf (qf32 + sf -> qf32).
 * T*L=1024 = 32 HVX vectors of 32 floats each; no tail needed.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define VFLOATS 32

void candidate_kernel(const float *x, float *y, int T, int L, float alpha) {
    int total = T * L;
    HVX_Vector valpha = Q6_V_vsplat_R(*(unsigned *)&alpha);

    int i;
    for (i = 0; i + VFLOATS <= total; i += VFLOATS) {
        HVX_Vector vx = *(const HVX_Vector *)(x + i);
        HVX_Vector vy = *(HVX_Vector *)(y + i);

        /* ax = alpha * x  (result in qf32) */
        HVX_Vector vax_qf = Q6_Vqf32_vmpy_VsfVsf(valpha, vx);

        /* y_new = ax + y  (qf32 + sf -> qf32, then convert to sf) */
        HVX_Vector vy_new = Q6_Vsf_equals_Vqf32(
                                Q6_Vqf32_vadd_Vqf32Vsf(vax_qf, vy));

        *(HVX_Vector *)(y + i) = vy_new;
    }
    /* scalar tail */
    for (; i < total; i++)
        y[i] = alpha * x[i] + y[i];
}
