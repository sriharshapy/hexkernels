#include <stdint.h>
#include <math.h>

static float v_sum_1[1];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  {
    float acc = 0;
    for (int r0 = 0; r0 < 48; r0++) {
      for (int r1 = 0; r1 < 128; r1++) {
        acc = (acc + v_args_0[r0*128 + r1*1]);
      }
    }
    v_sum_1[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_sum_1[i]; }
}
