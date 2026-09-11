/* sol_03: HVX-vectorized SGD momentum update.
 * v[i] = mu*v[i] + grad[i]   (fused multiply-add via qf32)
 * w[i] = w[i] - lr*v[i]      (fused multiply-subtract via qf32)
 * 32 floats per 128-byte HVX vector.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define VFLOATS 32

void candidate_kernel(float *w, float *v, const float *grad, int n,
                      float lr, float mu) {
    HVX_Vector vmu = Q6_V_vsplat_R(*(unsigned *)&mu);
    HVX_Vector vlr = Q6_V_vsplat_R(*(unsigned *)&lr);

    int nvec = n / VFLOATS;
    for (int i = 0; i < nvec; i++) {
        HVX_Vector vv    = *(HVX_Vector *)(v    + i * VFLOATS);
        HVX_Vector vg    = *(const HVX_Vector *)(grad + i * VFLOATS);
        HVX_Vector vw    = *(HVX_Vector *)(w    + i * VFLOATS);

        /* vi_new = mu * v + grad */
        HVX_Vector vmuv  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vmu, vv));
        HVX_Vector vv_new = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vmuv, vg));

        /* w_new = w - lr * v_new */
        HVX_Vector vlrv  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vlr, vv_new));
        HVX_Vector vw_new = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(vw, vlrv));

        *(HVX_Vector *)(v + i * VFLOATS) = vv_new;
        *(HVX_Vector *)(w + i * VFLOATS) = vw_new;
    }

    /* Scalar tail */
    for (int i = nvec * VFLOATS; i < n; i++) {
        float vi_new = mu * v[i] + grad[i];
        v[i] = vi_new;
        w[i] = w[i] - lr * vi_new;
    }
}
