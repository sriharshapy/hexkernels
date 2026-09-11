#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_args_0,
                                 const float *v_args_1,
                                 float *out0) {
  for (int i = 0; i < 1536; i += 32) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a, v_args_0 + i, sizeof(HVX_Vector));
    __builtin_memcpy(&b, v_args_1 + i, sizeof(HVX_Vector));

    const HVX_Vector r = Q6_Vsf_vmax_VsfVsf(a, b);
    __builtin_memcpy(out0 + i, &r, sizeof(HVX_Vector));
  }
}
