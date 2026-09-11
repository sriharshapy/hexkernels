#include <stdint.h>
#include <math.h>

static float v_asinh[2097152];
static unsigned char v_ne[2097152];
static int64_t v_sum_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, int64_t *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 2048; i1++) {
      v_asinh[i0*2048 + i1*1] = asinhf(v_xs_0[i0*2048 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 2048; i1++) {
      v_ne[i0*2048 + i1*1] = (v_asinh[i0*2048 + i1*1] != 0);
    }
  }
  {
    int64_t acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 2048; r1++) {
        acc = (acc + v_ne[r0*2048 + r1*1]);
      }
    }
    v_sum_1[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_sum_1[i]; }
}
