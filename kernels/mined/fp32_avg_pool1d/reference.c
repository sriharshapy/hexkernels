#include <stdint.h>
#include <math.h>

static float v_unsqueeze[4096];
static float v_avg_pool2d[2048];
static float v_squeeze[2048];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 4096; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 32; i3++) {
          float acc = 0;
          for (int k0 = 0; k0 < 1; k0++) {
            for (int k1 = 0; k1 < 2; k1++) {
              acc += v_unsqueeze[(i0)*4096 + (i1)*64 + (i2*1 + k0 - 0)*64 + (i3*2 + k1 - 0)];
            }
          }
          v_avg_pool2d[i1*32 + i3*1] = acc / (float)2;
        }
      }
    }
  }
  for (int i = 0; i < 2048; i++) v_squeeze[i] = v_avg_pool2d[i];
  for (int i = 0; i < 2048; i++) { out0[i] = v_squeeze[i]; }
}
