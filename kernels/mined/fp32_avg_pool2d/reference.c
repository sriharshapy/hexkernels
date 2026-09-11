#include <stdint.h>
#include <math.h>

static float v_avg_pool2d[1024];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          float acc = 0;
          for (int k0 = 0; k0 < 2; k0++) {
            for (int k1 = 0; k1 < 2; k1++) {
              acc += v_args_0[(i0)*4096 + (i1)*256 + (i2*2 + k0 - 0)*16 + (i3*2 + k1 - 0)];
            }
          }
          v_avg_pool2d[i1*64 + i2*8 + i3*1] = acc / (float)4;
        }
      }
    }
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v_avg_pool2d[i]; }
}
