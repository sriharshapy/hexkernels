#include <stdint.h>
#include <math.h>

static float v_atan[512];
static unsigned char v_ne[512];
static int64_t v_sum_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, int64_t *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_atan[i0*64 + i1*1] = atanf(v_xs_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_ne[i0*64 + i1*1] = (v_atan[i0*64 + i1*1] != 0);
    }
  }
  {
    int64_t acc = 0;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_ne[r0*64 + r1*1]);
      }
    }
    v_sum_1[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_sum_1[i]; }
}
