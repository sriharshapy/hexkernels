#include <stdint.h>
#include <math.h>

static float v_sub[1048576];
static float v_neg[1048576];
static float v_log1p[1048576];
static float v_full[1];
static float v_maximum[1048576];
static float v_mul[1048576];
static float v_log[1048576];
static float v_full_1[1];
static float v_maximum_1[1048576];
static float v_mul_1[1048576];
static float v_sub_1[1048576];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub[i0*1024 + i1*1] = (v_args_1[i0*1024 + i1*1] - 1);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_neg[i0*1024 + i1*1] = (-v_args_0[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_log1p[i0*1024 + i1*1] = log1pf(v_neg[i0*1024 + i1*1]);
    }
  }
  v_full[0] = (float)(-100);
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_maximum[i0*1024 + i1*1] = (v_log1p[i0*1024 + i1*1] > v_full[0] ? v_log1p[i0*1024 + i1*1] : v_full[0]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_sub[i0*1024 + i1*1] * v_maximum[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_log[i0*1024 + i1*1] = logf(v_args_0[i0*1024 + i1*1]);
    }
  }
  v_full_1[0] = (float)(-100);
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_maximum_1[i0*1024 + i1*1] = (v_log[i0*1024 + i1*1] > v_full_1[0] ? v_log[i0*1024 + i1*1] : v_full_1[0]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_1[i0*1024 + i1*1] = (v_args_1[i0*1024 + i1*1] * v_maximum_1[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub_1[i0*1024 + i1*1] = (v_mul[i0*1024 + i1*1] - v_mul_1[i0*1024 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc + v_sub_1[r0*1024 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)1048576);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
