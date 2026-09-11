#include <stdint.h>
#include <math.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_xs_0,
                                 const float *v_xs_1,
                                 float *out0) {
  alignas(128) float squared[512];

  for (int i = 0; i < 512; i += 32) {
    HVX_Vector x =
        Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(v_xs_0 + i));
    HVX_Vector y =
        Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(v_xs_1 + i));
    HVX_Vector d = Q6_Vsf_vsub_VsfVsf(x, y);
    HVX_Vector q = Q6_Vsf_vmpy_VsfVsf(d, d);
    Q6_V_vstu_AV(reinterpret_cast<HVX_Vector *>(squared + i), q);
  }

  volatile float acc = 0.0f;
#pragma clang loop vectorize(disable)
#pragma clang loop interleave(disable)
  for (int i = 0; i < 512; ++i) {
    acc = acc + squared[i];
  }

  float mean = static_cast<float>(acc) / 512.0f;
  out0[0] = (mean < 0.0f) ? -mean : mean;
}
