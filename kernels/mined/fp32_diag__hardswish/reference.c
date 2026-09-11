#include <stdint.h>
#include <math.h>

static float v_diagonal[1024];
static float v_clone[1024];
static float v_add[1024];
static float v_clamp[1024];
static float v_clamp_1[1024];
static float v_mul[1024];
static float v_div[1024];

extern "C" void candidate_kernel(const float *v_xs_0, float *out0) {
  for (int i = 0; i < 1024; i++) v_diagonal[i] = v_xs_0[(i + 0)*2048 + (i + 0)*1];
  for (int i0 = 0; i0 < 1024; i0++) {
    v_clone[i0*1] = v_diagonal[i0*1];
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_add[i0*1] = (v_clone[i0*1] + 3);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_clamp[i0*1] = (v_add[i0*1] < 0 ? 0 : v_add[i0*1]);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_clamp_1[i0*1] = (v_clamp[i0*1] > 6 ? 6 : v_clamp[i0*1]);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_mul[i0*1] = (v_clone[i0*1] * v_clamp_1[i0*1]);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_div[i0*1] = (v_mul[i0*1] / 6);
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v_div[i]; }
}
