#include <stdint.h>
#include <math.h>

static float v_diagonal[1024];
static float v_clone[1024];
static float v_sum_1[1];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 1024; i++) v_diagonal[i] = v_args_0[(i + 0)*2048 + (i + 0)*1];
  for (int i0 = 0; i0 < 1024; i0++) {
    v_clone[i0*1] = v_diagonal[i0*1];
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      acc = (acc + v_clone[r0*1]);
    }
    v_sum_1[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_sum_1[i]; }
}
