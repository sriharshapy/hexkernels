#include <stdint.h>
#include <math.h>

static float v_slice_1[6096];
static float v_slice_2[6096];
static float v_sub[6096];

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
      v_sub[i0*127 + i1*1] = (v_slice_2[i0*127 + i1*1] - v_slice_1[i0*127 + i1*1]);
    }
  }
  for (int i = 0; i < 6096; i++) { out0[i] = v_sub[i]; }
}
