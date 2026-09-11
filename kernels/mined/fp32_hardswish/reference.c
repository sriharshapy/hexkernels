#include <stdint.h>
#include <math.h>

static float v_add[6144];
static float v_clamp[6144];
static float v_clamp_1[6144];
static float v_mul[6144];
static float v_div[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_add[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] + 3);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_clamp[i0*128 + i1*1] = (v_add[i0*128 + i1*1] < 0 ? 0 : v_add[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_clamp_1[i0*128 + i1*1] = (v_clamp[i0*128 + i1*1] > 6 ? 6 : v_clamp[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_mul[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] * v_clamp_1[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_div[i0*128 + i1*1] = (v_mul[i0*128 + i1*1] / 6);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_div[i]; }
}
