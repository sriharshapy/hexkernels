#include <stdint.h>
#include <math.h>

static float v_sub[512];
static float v_pow_1[512];
static float v_mean[1];
static float v_abs_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub[i0*64 + i1*1] = (v_xs_0[i0*64 + i1*1] - v_xs_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_pow_1[i0*64 + i1*1] = powf(v_sub[i0*64 + i1*1], 2);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_pow_1[r0*64 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)512);
  }
  v_abs_1[0] = (v_mean[0] < 0 ? -v_mean[0] : v_mean[0]);
  for (int i = 0; i < 1; i++) { out0[i] = v_abs_1[i]; }
}
