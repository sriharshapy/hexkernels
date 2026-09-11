#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  static const uint32_t neg_inf[32] = {
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u,
      0xff800000u, 0xff800000u, 0xff800000u, 0xff800000u
  };
  static unsigned char mask[128];

  const HVX_Vector vneg =
      Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(neg_inf));

  for (int base = 0; base < 512; base += 32) {
    const HVX_Vector vin =
        Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(v_args_0 + base));
    const HVX_VectorPred eq = Q6_Q_vcmp_eq_VwVw(vin, vneg);
    const HVX_Vector selected =
        Q6_V_vmux_QVV(eq, Q6_V_vsplatb(1), Q6_V_vzero());

    Q6_V_vstu_AV(reinterpret_cast<HVX_Vector *>(mask), selected);

    for (int i = 0; i < 32; ++i)
      out0[base + i] = mask[i * 4];
  }
}
