#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

// v75 HVX vectors are 128 bytes wide, as specified by the target architecture.
extern "C" void candidate_kernel(const int32_t *v_args_0,
                                 const int32_t *v_args_1,
                                 int32_t *out0) {
  constexpr int kElementsPerVector = sizeof(HVX_Vector) / sizeof(int32_t);

  for (int i = 0; i < 1536; i += kElementsPerVector) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a, v_args_0 + i, sizeof(HVX_Vector));
    __builtin_memcpy(&b, v_args_1 + i, sizeof(HVX_Vector));

    const HVX_Vector result = Q6_V_vor_VV(a, b);
    __builtin_memcpy(out0 + i, &result, sizeof(HVX_Vector));
  }
}
