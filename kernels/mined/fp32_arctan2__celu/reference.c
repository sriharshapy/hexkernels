#include <stdint.h>
#include <math.h>

static float v_atan2[98304];
static float v_expm1[98304];
static unsigned char v_gt[98304];
static float v_where[98304];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_atan2[i0*384 + i1*1] = atan2f(v_xs_0[i0*384 + i1*1], v_xs_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_expm1[i0*384 + i1*1] = expm1f(v_atan2[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_gt[i0*384 + i1*1] = (v_atan2[i0*384 + i1*1] > 0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where[i0*384 + i1*1] = (v_gt[i0*384 + i1*1] ? v_atan2[i0*384 + i1*1] : v_expm1[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_where[i]; }
}
