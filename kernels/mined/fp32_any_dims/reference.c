#include <stdint.h>
#include <math.h>

static unsigned char v_any_1[1];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 2048; r1++) {
        acc = (acc || (v_args_0[r0*2048 + r1*1] != 0));
      }
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_any_1[i]; }
}
