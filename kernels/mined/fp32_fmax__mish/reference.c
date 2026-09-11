#include <stdint.h>
#include <math.h>

static float v_fmax[98304];
static float v_exp[98304];
static float v_log1p[98304];
static unsigned char v_gt[98304];
static float v_where[98304];
static float v_tanh[98304];
static float v_mul[98304];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_fmax[i0*384 + i1*1] = fmaxf(v_xs_0[i0*384 + i1*1], v_xs_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_exp[i0*384 + i1*1] = expf(v_fmax[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_log1p[i0*384 + i1*1] = log1pf(v_exp[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_gt[i0*384 + i1*1] = (v_fmax[i0*384 + i1*1] > 20);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where[i0*384 + i1*1] = (v_gt[i0*384 + i1*1] ? v_fmax[i0*384 + i1*1] : v_log1p[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_tanh[i0*384 + i1*1] = tanhf(v_where[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_fmax[i0*384 + i1*1] * v_tanh[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_mul[i]; }
}
