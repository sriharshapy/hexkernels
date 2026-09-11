#include <stdint.h>
#include <math.h>

static float v_erf[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_erf[i0*64 + i1*1] = erff(v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_erf[i]; }
}
