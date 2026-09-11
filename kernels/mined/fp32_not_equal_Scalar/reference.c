#include <stdint.h>
#include <math.h>

static unsigned char v_ne[512];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_ne[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] != 2.0f);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_ne[i]; }
}
