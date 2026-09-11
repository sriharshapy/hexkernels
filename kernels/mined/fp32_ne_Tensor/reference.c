#include <stdint.h>
#include <math.h>

static unsigned char v_ne[786432];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 768; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_ne[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] != v_args_1[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_ne[i]; }
}
