#include <stdint.h>
#include <math.h>

static float v_mean[4096];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 64; i1++) {
    for (int i2 = 0; i2 < 64; i2++) {
      {
        float acc = 0;
        for (int r0 = 0; r0 < 1; r0++) {
          acc = (acc + v_args_0[i1*64 + i2*1]);
        }
        v_mean[i1*64 + i2*1] = (acc / (float)1);
      }
    }
  }
  for (int i = 0; i < 4096; i++) { out0[i] = v_mean[i]; }
}
