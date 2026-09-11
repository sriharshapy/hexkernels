#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

// v75 HVX provides 128-byte vectors, as specified by the target hardware.
extern "C" void candidate_kernel(const float *v_args_0,
                                 const float *v_args_1,
                                 float *out0) {
  const HVX_Vector one = Q6_V_vsplat_R(0x3f800000);

  for (int i = 0; i < 512; i += 32) {
    const HVX_Vector self =
        Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(v_args_0 + i));
    const HVX_Vector end =
        Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(v_args_1 + i));

    const HVX_Vector difference =
        Q6_Vqf32_vsub_Vqf32Vqf32(end, self);
    const HVX_Vector scaled =
        Q6_Vqf32_vmpy_Vqf32Vqf32(difference, one);
    const HVX_Vector result =
        Q6_Vqf32_vadd_Vqf32Vqf32(end, scaled);

    Q6_V_vstu_AV(reinterpret_cast<HVX_Vector *>(out0 + i), result);
  }
}
