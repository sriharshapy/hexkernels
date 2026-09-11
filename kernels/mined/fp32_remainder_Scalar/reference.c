#include <stdint.h>
#include <math.h>

static float v_remainder[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_remainder[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] - floorf(v_args_0[i0*1024 + i1*1] / 2.0f) * 2.0f);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_remainder[i]; }
}
