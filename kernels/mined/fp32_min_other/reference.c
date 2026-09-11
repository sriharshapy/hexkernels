#include <stdint.h>
#include <math.h>

static float v_minimum[98304];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_minimum[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] < v_args_1[i0*384 + i1*1] ? v_args_0[i0*384 + i1*1] : v_args_1[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_minimum[i]; }
}
