#include <stdint.h>
#include <math.h>

static float v_sub[1048576];
static float v_abs_1[1048576];
static unsigned char v_lt[1048576];
static float v_mul[1048576];
static float v_mul_1[1048576];
static float v_sub_1[1048576];
static float v_mul_2[1048576];
static float v_where[1048576];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] - v_args_1[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_abs_1[i0*1024 + i1*1] = (v_sub[i0*1024 + i1*1] < 0 ? -v_sub[i0*1024 + i1*1] : v_sub[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_lt[i0*1024 + i1*1] = (v_abs_1[i0*1024 + i1*1] < 1.0f);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_abs_1[i0*1024 + i1*1] * 0.5f);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_1[i0*1024 + i1*1] = (v_mul[i0*1024 + i1*1] * v_abs_1[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub_1[i0*1024 + i1*1] = (v_abs_1[i0*1024 + i1*1] - 0.5f);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_2[i0*1024 + i1*1] = (v_sub_1[i0*1024 + i1*1] * 1.0f);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_lt[i0*1024 + i1*1] ? v_mul_1[i0*1024 + i1*1] : v_mul_2[i0*1024 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc + v_where[r0*1024 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)1048576);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
