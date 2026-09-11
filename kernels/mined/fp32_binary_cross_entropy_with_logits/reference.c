#include <stdint.h>
#include <math.h>

static float v_sub[512];
static float v_mul[512];
static float v_full[1];
static float v_minimum[512];
static float v_abs_1[512];
static float v_neg[512];
static float v_exp[512];
static float v_log1p[512];
static float v_sub_1[512];
static float v_sub_2[512];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub[i0*64 + i1*1] = (1 - v_args_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_sub[i0*64 + i1*1] * v_args_0[i0*64 + i1*1]);
    }
  }
  v_full[0] = (float)(0);
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_minimum[i0*64 + i1*1] = (v_full[0] < v_args_0[i0*64 + i1*1] ? v_full[0] : v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_abs_1[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] < 0 ? -v_args_0[i0*64 + i1*1] : v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_neg[i0*64 + i1*1] = (-v_abs_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_exp[i0*64 + i1*1] = expf(v_neg[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_log1p[i0*64 + i1*1] = log1pf(v_exp[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_1[i0*64 + i1*1] = (v_minimum[i0*64 + i1*1] - v_log1p[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_2[i0*64 + i1*1] = (v_mul[i0*64 + i1*1] - v_sub_1[i0*64 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_sub_2[r0*64 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)512);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
