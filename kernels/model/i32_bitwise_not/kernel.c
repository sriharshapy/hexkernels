#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const int32_t *v_args_0, int32_t *out0) {
  const HVX_Vector all_ones = Q6_V_vsplat_R(-1);

  for (int i = 0; i < 6144; i += 32) {
    HVX_Vector input;
    __builtin_memcpy(&input, v_args_0 + i, sizeof(HVX_Vector));

    HVX_Vector result = Q6_V_vxor_VV(input, all_ones);
    __builtin_memcpy(out0 + i, &result, sizeof(HVX_Vector));
  }
}
