#include <stdint.h>
#include <math.h>

static float v_remainder[1536];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 24; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_remainder[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] - floorf(v_args_0[i0*64 + i1*1] / v_args_1[i0*64 + i1*1]) * v_args_1[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 1536; i++) { out0[i] = v_remainder[i]; }
}
