#include <stdint.h>
#include <math.h>

static float v_log2[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_log2[i0*128 + i1*1] = log2f(v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_log2[i]; }
}
