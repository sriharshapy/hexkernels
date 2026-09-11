#include <stdint.h>
#include <math.h>

static float v_avg_pool3d[65536];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            float acc = 0;
            for (int k0 = 0; k0 < 2; k0++) {
              for (int k1 = 0; k1 < 2; k1++) {
                for (int k2 = 0; k2 < 2; k2++) {
                  acc += v_args_0[(i0)*524288 + (i1)*32768 + (i2*2 + k0 - 0)*1024 + (i3*2 + k1 - 0)*32 + (i4*2 + k2 - 0)];
                }
              }
            }
            v_avg_pool3d[i1*4096 + i2*256 + i3*16 + i4*1] = acc / (float)8;
          }
        }
      }
    }
  }
  for (int i = 0; i < 65536; i++) { out0[i] = v_avg_pool3d[i]; }
}
