#include <stdint.h>
#include <math.h>

static unsigned char v_eq[98304];
static float v_log[98304];
static float v_mul[98304];
static float v_scalar_tensor[1];
static float v_where[98304];
static unsigned char v_isnan[98304];
static float v_scalar_tensor_1[1];
static float v_where_1[98304];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] == 0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_log[i0*384 + i1*1] = logf(v_args_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] * v_log[i0*384 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where[i0*384 + i1*1] = (v_eq[i0*384 + i1*1] ? v_scalar_tensor[0] : v_mul[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_isnan[i0*384 + i1*1] = (v_args_1[i0*384 + i1*1] != v_args_1[i0*384 + i1*1]);
    }
  }
  v_scalar_tensor_1[0] = (float)(NAN);
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_where_1[i0*384 + i1*1] = (v_isnan[i0*384 + i1*1] ? v_scalar_tensor_1[0] : v_where[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_where_1[i]; }
}
