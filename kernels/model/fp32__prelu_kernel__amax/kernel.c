#include <math.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_xs_0,
                                 const float *v_xs_1,
                                 float *out0) {
  // HVX v75 vectors are 128 bytes wide, i.e. 32 float32 elements.
  alignas(128) float tile[32];

  float acc = -INFINITY;
  const HVX_Vector vzero = Q6_V_vzero();

  for (int row = 0; row < 8; ++row) {
    const float *x0 = v_xs_0 + row * 64;
    const float *x1 = v_xs_1 + row * 64;

    for (int col = 0; col < 64; col += 32) {
      const HVX_Vector a =
          Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(x0 + col));
      const HVX_Vector b =
          Q6_V_vldu_A(reinterpret_cast<const HVX_Vector *>(x1 + col));

      const HVX_Vector product = Q6_Vsf_vmpy_VsfVsf(b, a);
      const HVX_VectorPred positive = Q6_Q_vcmp_gt_VsfVsf(a, vzero);
      const HVX_Vector selected = Q6_V_vmux_QVV(positive, a, product);

      Q6_V_vstu_AV(reinterpret_cast<HVX_Vector *>(tile), selected);

      for (int lane = 0; lane < 32; ++lane) {
        const float value = tile[lane];
        acc = (value > acc) ? value : acc;
      }
    }
  }

  out0[0] = acc;
}
