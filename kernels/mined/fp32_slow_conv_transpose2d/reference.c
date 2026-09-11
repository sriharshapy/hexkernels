#include <stdint.h>
#include <math.h>

static float v_slow_conv_transpose2d[324];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 9; i2++) {
        for (int i3 = 0; i3 < 9; i3++) {
          float acc = 0;
          for (int r0 = 0; r0 < 4; r0++) {
            for (int r1 = 0; r1 < 2; r1++) {
              int th = i2 + 0 - r1*1;
              if (th < 0 || (th % 1) != 0) continue;
              int ih = th / 1;
              if (ih >= 8) continue;
              for (int r2 = 0; r2 < 2; r2++) {
                int tw = i3 + 0 - r2*1;
                if (tw < 0 || (tw % 1) != 0) continue;
                int iw = tw / 1;
                if (iw >= 8) continue;
                acc += v_args_0[i0*256 + r0*64 + ih*8 + iw]
                  * v_args_1[r0*16 + i1*4 + r1*2 + r2];
              }
            }
          }
          v_slow_conv_transpose2d[i0*324 + i1*81 + i2*9 + i3] = acc;
        }
      }
    }
  }
  for (int i = 0; i < 324; i++) { out0[i] = v_slow_conv_transpose2d[i]; }
}
