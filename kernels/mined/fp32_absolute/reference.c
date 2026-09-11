#include <stdint.h>
#include <math.h>

static float v_abs_1[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_abs_1[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] < 0 ? -v_args_0[i0*1024 + i1*1] : v_args_0[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_abs_1[i]; }
}
