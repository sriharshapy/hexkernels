#include <stdint.h>
#include <math.h>

static float v_full[1];
static float v_minimum[196608];
static float v_abs_1[196608];
static float v_neg[196608];
static float v_exp[196608];
static float v_log1p[196608];
static float v_sub[196608];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  v_full[0] = (float)(0);
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_minimum[i0*512 + i1*1] = (v_full[0] < v_args_0[i0*512 + i1*1] ? v_full[0] : v_args_0[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_abs_1[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] < 0 ? -v_args_0[i0*512 + i1*1] : v_args_0[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_neg[i0*512 + i1*1] = (-v_abs_1[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_exp[i0*512 + i1*1] = expf(v_neg[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_log1p[i0*512 + i1*1] = log1pf(v_exp[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub[i0*512 + i1*1] = (v_minimum[i0*512 + i1*1] - v_log1p[i0*512 + i1*1]);
    }
  }
  for (int i = 0; i < 196608; i++) { out0[i] = v_sub[i]; }
}
