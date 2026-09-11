#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1) {
  float scalar_best[64];
  int64_t scalar_arg[64];

  for (int i1 = 0; i1 < 64; ++i1) {
    float best = __builtin_inff();
    int64_t arg = 0;
    for (int s0 = 0; s0 < 8; ++s0) {
      const float x = v_args_0[s0 * 64 + i1];
      if (x < best) {
        best = x;
        arg = s0;
      }
      out1[s0 * 64 + i1] = arg;
    }
    scalar_best[i1] = best;
    scalar_arg[i1] = arg;
  }

  const HVX_Vector v_inf = Q6_V_vsplatw_R(0x7f800000);
  HVX_Vector v_best = v_inf;

  for (int s0 = 0; s0 < 8; ++s0) {
    const HVX_Vector v_x =
        Q6_V_vldu_A((const HVX_Vector *)(v_args_0 + s0 * 64));

    // For finite values, best > x is equivalent to x < best.
    // Both comparisons are false for NaNs, matching the scalar reference.
    const HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(v_best, v_x);
    v_best = Q6_V_vmux_QVV(pred, v_x, v_best);

    Q6_V_vstu_AV((HVX_Vector *)(out0 + s0 * 64), v_best);
  }
}
