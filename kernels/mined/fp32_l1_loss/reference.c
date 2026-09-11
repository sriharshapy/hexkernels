#include <stdint.h>
#include <math.h>

static float v_sub[196608];
static float v_abs_1[196608];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] - v_args_1[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_abs_1[i0*512 + i1*1] = (v_sub[i0*512 + i1*1] < 0 ? -v_sub[i0*512 + i1*1] : v_sub[i0*512 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 384; r0++) {
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_abs_1[r0*512 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)196608);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
