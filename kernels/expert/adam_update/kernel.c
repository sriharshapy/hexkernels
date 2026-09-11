/* sol_03: HVX-vectorized Adam update.
 * m, v, w updates via qf32 fused multiply-adds.
 * sqrtf / division done scalar (no HVX rsqrt intrinsic).
 * 32 floats per 128-byte HVX vector; n=256 is exactly 8 vectors.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define VFLOATS 32

void candidate_kernel(float *w, float *m, float *v, const float *grad, int n,
                      float lr, float b1, float b2, float eps, int t) {
    float bc1  = 1.0f - powf(b1, (float)t);
    float bc2  = 1.0f - powf(b2, (float)t);
    float b1c  = 1.0f - b1;
    float b2c  = 1.0f - b2;
    float ibc1 = 1.0f / bc1;
    float ibc2 = 1.0f / bc2;

    HVX_Vector vb1  = Q6_V_vsplat_R(*(unsigned *)&b1);
    HVX_Vector vb2  = Q6_V_vsplat_R(*(unsigned *)&b2);
    HVX_Vector vb1c = Q6_V_vsplat_R(*(unsigned *)&b1c);
    HVX_Vector vb2c = Q6_V_vsplat_R(*(unsigned *)&b2c);

    /* staging buffer for scalar sqrt/div pass */
    float mbuf[VFLOATS] __attribute__((aligned(128)));
    float vbuf[VFLOATS] __attribute__((aligned(128)));
    float wbuf[VFLOATS] __attribute__((aligned(128)));

    int nvec = n / VFLOATS;
    for (int i = 0; i < nvec; i++) {
        HVX_Vector vm = *(HVX_Vector *)(m    + i * VFLOATS);
        HVX_Vector vv = *(HVX_Vector *)(v    + i * VFLOATS);
        HVX_Vector vw = *(HVX_Vector *)(w    + i * VFLOATS);
        HVX_Vector vg = *(const HVX_Vector *)(grad + i * VFLOATS);

        /* m_new = b1*m + (1-b1)*g */
        HVX_Vector vb1m   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vb1, vm));
        HVX_Vector vb1cg  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vb1c, vg));
        HVX_Vector vm_new = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vb1m, vb1cg));

        /* g2 = g * g */
        HVX_Vector vg2    = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vg, vg));

        /* v_new = b2*v + (1-b2)*g^2 */
        HVX_Vector vb2v   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vb2, vv));
        HVX_Vector vb2cg2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vb2c, vg2));
        HVX_Vector vv_new = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vb2v, vb2cg2));

        /* store m and v (updated) */
        *(HVX_Vector *)(m + i * VFLOATS) = vm_new;
        *(HVX_Vector *)(v + i * VFLOATS) = vv_new;

        /* scalar pass: mhat, vhat, sqrt, w update */
        *(HVX_Vector *)mbuf = vm_new;
        *(HVX_Vector *)vbuf = vv_new;
        *(HVX_Vector *)wbuf = vw;

        for (int j = 0; j < VFLOATS; j++) {
            float mhat = mbuf[j] * ibc1;
            float vhat = vbuf[j] * ibc2;
            wbuf[j] = wbuf[j] - lr * mhat / (sqrtf(vhat) + eps);
        }
        *(HVX_Vector *)(w + i * VFLOATS) = *(HVX_Vector *)wbuf;
    }

    /* Scalar tail */
    for (int i = nvec * VFLOATS; i < n; i++) {
        float gi   = grad[i];
        float mi_n = b1 * m[i] + b1c * gi;
        float vi_n = b2 * v[i] + b2c * gi * gi;
        m[i] = mi_n;
        v[i] = vi_n;
        float mhat = mi_n * ibc1;
        float vhat = vi_n * ibc2;
        w[i] = w[i] - lr * mhat / (sqrtf(vhat) + eps);
    }
}
