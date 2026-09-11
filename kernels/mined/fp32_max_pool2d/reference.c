#include <stdint.h>
#include <math.h>

static float v_max_pool2d_with_indices_0[64];
static int64_t v_max_pool2d_with_indices_1[64];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 4; i2++) {
        for (int i3 = 0; i3 < 4; i3++) {
          float best = (float)-INFINITY;
          int64_t arg = 0;
          for (int k0 = 0; k0 < 2; k0++) {
            for (int k1 = 0; k1 < 2; k1++) {
              if (v_args_0[(i0)*256 + (i1)*64 + (i2*2 + k0 - 0)*8 + (i3*2 + k1 - 0)] > best) { best = v_args_0[(i0)*256 + (i1)*64 + (i2*2 + k0 - 0)*8 + (i3*2 + k1 - 0)]; arg = (int64_t)((i2*2 + k0 - 0)*8 + (i3*2 + k1 - 0)); }
            }
          }
          v_max_pool2d_with_indices_0[i1*16 + i2*4 + i3*1] = best;
          v_max_pool2d_with_indices_1[i1*16 + i2*4 + i3*1] = arg;
        }
      }
    }
  }
  for (int i = 0; i < 64; i++) { out0[i] = v_max_pool2d_with_indices_0[i]; }
}
