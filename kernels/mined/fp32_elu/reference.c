#include <stdint.h>
#include <math.h>

static unsigned char v_gt[1310720];
static float v_mul[1310720];
static float v_mul_1[1310720];
static float v_expm1[1310720];
static float v_mul_2[1310720];
static float v_where[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_gt[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] > 0);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] * 1.0f);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_1[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] * 1.0f);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_expm1[i0*1024 + i1*1] = expm1f(v_mul_1[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul_2[i0*1024 + i1*1] = (v_expm1[i0*1024 + i1*1] * 1.0f);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_gt[i0*1024 + i1*1] ? v_mul[i0*1024 + i1*1] : v_mul_2[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_where[i]; }
}
