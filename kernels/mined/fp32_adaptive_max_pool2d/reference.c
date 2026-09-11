#include <stdint.h>
#include <math.h>

static float v_adaptive_max_pool2d_0[1024];
static int64_t v_adaptive_max_pool2d_1[1024];

extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          const int s0 = (i2 * 16) / 8;
          const int e0 = ((i2 + 1) * 16 + 8 - 1) / 8;
          const int s1 = (i3 * 16) / 8;
          const int e1 = ((i3 + 1) * 16 + 8 - 1) / 8;
          float best = (float)-INFINITY;
          int64_t arg = 0;
          for (int k0 = s0; k0 < e0; k0++) {
            for (int k1 = s1; k1 < e1; k1++) {
              if (v_args_0[(i0)*4096 + (i1)*256 + (k0)*16 + (k1)] > best) { best = v_args_0[(i0)*4096 + (i1)*256 + (k0)*16 + (k1)]; arg = (int64_t)((k0)*16 + (k1)); }
            }
          }
          v_adaptive_max_pool2d_0[i1*64 + i2*8 + i3*1] = best;
          v_adaptive_max_pool2d_1[i1*64 + i2*8 + i3*1] = arg;
        }
      }
    }
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v_adaptive_max_pool2d_0[i]; }
  for (int i = 0; i < 1024; i++) { out1[i] = v_adaptive_max_pool2d_1[i]; }
}
