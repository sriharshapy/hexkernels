#include <stdint.h>
#include <math.h>

static float v_prod[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 512; i1++) {
    {
      float acc = 1;
      for (int r0 = 0; r0 < 512; r0++) {
        acc = (acc * v_args_0[r0*512 + i1*1]);
      }
      v_prod[i1*1] = acc;
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_prod[i]; }
}
