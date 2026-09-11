#include <stdint.h>
#include <math.h>

static float v__adaptive_avg_pool3d[4096];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          for (int i4 = 0; i4 < 8; i4++) {
            const int s0 = (i2 * 16) / 8;
            const int e0 = ((i2 + 1) * 16 + 8 - 1) / 8;
            const int s1 = (i3 * 16) / 8;
            const int e1 = ((i3 + 1) * 16 + 8 - 1) / 8;
            const int s2 = (i4 * 16) / 8;
            const int e2 = ((i4 + 1) * 16 + 8 - 1) / 8;
            float acc = 0;
            int cnt = 0;
            for (int k0 = s0; k0 < e0; k0++) {
              for (int k1 = s1; k1 < e1; k1++) {
                for (int k2 = s2; k2 < e2; k2++) {
                  acc += v_args_0[(i0)*32768 + (i1)*4096 + (k0)*256 + (k1)*16 + (k2)];
                  cnt++;
                }
              }
            }
            v__adaptive_avg_pool3d[i1*512 + i2*64 + i3*8 + i4*1] = acc / (float)cnt;
          }
        }
      }
    }
  }
  for (int i = 0; i < 4096; i++) { out0[i] = v__adaptive_avg_pool3d[i]; }
}
