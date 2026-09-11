#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_args_0,
                                 const float *v_args_1,
                                 float *out0) {
  for (int i = 0; i < 1536; i += 32) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a, v_args_0 + i, sizeof(HVX_Vector));
    __builtin_memcpy(&b, v_args_1 + i, sizeof(HVX_Vector));

    const HVX_Vector product =
        Q6_Vsf_equals_Vqf32(
            Q6_Vqf32_vmpy_Vqf32Vqf32(a, b));

    const HVX_VectorPred positive =
        Q6_Q_vcmp_gt_VsfVsf(a, Q6_V_vzero());

    const HVX_Vector result = Q6_V_vmux_QVV(positive, a, product);

    __builtin_memcpy(out0 + i, &result, sizeof(HVX_Vector));
  }
}
