#include <stdint.h>
#include <math.h>

static float v_unsqueeze[32];
static float v__adaptive_avg_pool2d[16];
static float v_squeeze[16];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 32; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        for (int i3 = 0; i3 < 4; i3++) {
          const int s0 = (i2 * 1) / 1;
          const int e0 = ((i2 + 1) * 1 + 1 - 1) / 1;
          const int s1 = (i3 * 8) / 4;
          const int e1 = ((i3 + 1) * 8 + 4 - 1) / 4;
          float acc = 0;
          int cnt = 0;
          for (int k0 = s0; k0 < e0; k0++) {
            for (int k1 = s1; k1 < e1; k1++) {
              acc += v_unsqueeze[(i0)*32 + (i1)*8 + (k0)*8 + (k1)];
              cnt++;
            }
          }
          v__adaptive_avg_pool2d[i1*4 + i3*1] = acc / (float)cnt;
        }
      }
    }
  }
  for (int i = 0; i < 16; i++) v_squeeze[i] = v__adaptive_avg_pool2d[i];
  for (int i = 0; i < 16; i++) { out0[i] = v_squeeze[i]; }
}
