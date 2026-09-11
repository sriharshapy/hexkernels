#include <stdint.h>
#include <math.h>

static float v_slice_1[6096];
static float v_slice_2[6096];
static float v_add[6096];
static float v_mul[6096];
static float v_cumsum[6096];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 127; i1++) {
      v_slice_1[i0*127 + i1*1] = v_args_0[i0*128 + i1];
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 127; i1++) {
      v_slice_2[i0*127 + i1*1] = v_args_0[i0*128 + (1 + i1 * 1)];
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 127; i1++) {
      v_add[i0*127 + i1*1] = (v_slice_1[i0*127 + i1*1] + v_slice_2[i0*127 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 127; i1++) {
      v_mul[i0*127 + i1*1] = (v_add[i0*127 + i1*1] * 0.5f);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    float acc = 0;
    for (int s1 = 0; s1 < 127; s1++) {
      acc = (acc + v_mul[i0*127 + s1*1]);
      v_cumsum[i0*127 + s1*1] = (float)acc;
    }
  }
  for (int i = 0; i < 6096; i++) { out0[i] = v_cumsum[i]; }
}
