#include <stdint.h>
#include <math.h>

static float v_prod[1];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  {
    float acc = 1;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc * v_args_0[r0*64 + r1*1]);
      }
    }
    v_prod[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_prod[i]; }
}
