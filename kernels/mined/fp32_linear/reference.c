#include <stdint.h>
#include <math.h>

static float v_permute[786432];
static float v_mm[589824];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 768; i1++) {
      v_permute[i0*768 + i1*1] = v_args_1[i1*1024 + i0];
    }
  }
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 768; i1++) {
      float acc = 0;
      for (int k = 0; k < 1024; k++) {
        acc = acc + v_args_0[i0*1024 + k] * v_permute[k*768 + i1];
      }
    v_mm[i0*768 + i1] = acc;
    }
  }
  for (int i = 0; i < 589824; i++) { out0[i] = v_mm[i]; }
}
