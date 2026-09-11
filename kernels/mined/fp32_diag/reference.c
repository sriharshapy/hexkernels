#include <stdint.h>
#include <math.h>

static float v_diagonal[512];
static float v_clone[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 512; i++) v_diagonal[i] = v_args_0[(i + 0)*512 + (i + 0)*1];
  for (int i0 = 0; i0 < 512; i0++) {
    v_clone[i0*1] = v_diagonal[i0*1];
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_clone[i]; }
}
