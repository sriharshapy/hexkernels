#include <stdint.h>
#include <math.h>

static float v_slow_conv3d_forward[8001504];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 32; i1++) {
      for (int i2 = 0; i2 < 63; i2++) {
        for (int i3 = 0; i3 < 63; i3++) {
          for (int i4 = 0; i4 < 63; i4++) {
            float acc = 0;
            for (int r0 = 0; r0 < 32; r0++) {
              for (int r1 = 0; r1 < 2; r1++) {
                int id = i2*1 + r1 - 0;
                if (id < 0 || id >= 64) continue;
                for (int r2 = 0; r2 < 2; r2++) {
                  int ih = i3*1 + r2 - 0;
                  if (ih < 0 || ih >= 64) continue;
                  for (int r3 = 0; r3 < 2; r3++) {
                    int iw = i4*1 + r3 - 0;
                    if (iw < 0 || iw >= 64) continue;
                    acc += v_args_0[i0*8388608 + r0*262144 + id*4096 + ih*64 + iw]
                      * v_args_1[i1*256 + r0*8 + r1*4 + r2*2 + r3];
                  }
                }
              }
            }
            v_slow_conv3d_forward[i0*8001504 + i1*250047 + i2*3969 + i3*63 + i4] = acc;
          }
        }
      }
    }
  }
  for (int i = 0; i < 8001504; i++) { out0[i] = v_slow_conv3d_forward[i]; }
}
