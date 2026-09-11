#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_args_0,
                                 const float *v_args_1,
                                 float *out0) {
  constexpr int kElementsPerVector = 32;
  constexpr int kElementCount = 512;

  for (int i = 0; i < kElementCount; i += kElementsPerVector) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a, v_args_0 + i, sizeof(HVX_Vector));
    __builtin_memcpy(&b, v_args_1 + i, sizeof(HVX_Vector));

    const HVX_Vector result =
        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));

    __builtin_memcpy(out0 + i, &result, sizeof(HVX_Vector));
  }
}
