#include <stdint.h>
#include <math.h>

static unsigned char v_isnan[6144];
static unsigned char v_gt[6144];
static float v_neg[6144];
static float v_log[6144];
static float v_mul[6144];
static unsigned char v_eq[6144];
static float v_scalar_tensor[1];
static float v_scalar_tensor_1[1];
static float v_where[6144];
static float v_where_1[6144];
static float v_where_2[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_isnan[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] != v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_gt[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] > 0);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_neg[i0*128 + i1*1] = (-v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_log[i0*128 + i1*1] = logf(v_args_0[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_mul[i0*128 + i1*1] = (v_neg[i0*128 + i1*1] * v_log[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_eq[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] == 0);
    }
  }
  v_scalar_tensor[0] = (float)((-INFINITY));
  v_scalar_tensor_1[0] = (float)(0);
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_where[i0*128 + i1*1] = (v_eq[i0*128 + i1*1] ? v_scalar_tensor_1[0] : v_scalar_tensor[0]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_where_1[i0*128 + i1*1] = (v_gt[i0*128 + i1*1] ? v_mul[i0*128 + i1*1] : v_where[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_where_2[i0*128 + i1*1] = (v_isnan[i0*128 + i1*1] ? v_args_0[i0*128 + i1*1] : v_where_1[i0*128 + i1*1]);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_where_2[i]; }
}
