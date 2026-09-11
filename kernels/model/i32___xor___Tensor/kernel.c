#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const int32_t *v_args_0,
                                 const int32_t *v_args_1,
                                 int32_t *out0) {
  constexpr int kElements = 768 * 1024;
  constexpr int kBytesPerVector = 128;
  constexpr int kElementsPerVector = kBytesPerVector / sizeof(int32_t);

  for (int i = 0; i < kElements; i += kElementsPerVector) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a, v_args_0 + i, kBytesPerVector);
    __builtin_memcpy(&b, v_args_1 + i, kBytesPerVector);

    HVX_Vector result = Q6_V_vxor_VV(a, b);
    __builtin_memcpy(out0 + i, &result, kBytesPerVector);
  }
}
