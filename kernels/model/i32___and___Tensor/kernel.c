#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const int32_t *v_args_0,
                                 const int32_t *v_args_1,
                                 int32_t *out0) {
  constexpr int kVectorBytes = 128;
  constexpr int kTotalBytes = 512 * static_cast<int>(sizeof(int32_t));

  for (int offset = 0; offset < kTotalBytes; offset += kVectorBytes) {
    HVX_Vector a;
    HVX_Vector b;

    __builtin_memcpy(&a,
                     reinterpret_cast<const char *>(v_args_0) + offset,
                     kVectorBytes);
    __builtin_memcpy(&b,
                     reinterpret_cast<const char *>(v_args_1) + offset,
                     kVectorBytes);

    const HVX_Vector result = Q6_V_vand_VV(a, b);

    __builtin_memcpy(reinterpret_cast<char *>(out0) + offset,
                     &result,
                     kVectorBytes);
  }
}
