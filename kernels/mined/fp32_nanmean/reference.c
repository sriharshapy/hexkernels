#include <stdint.h>
#include <math.h>

static float v_alias[6144];
static unsigned char v_ne[6144];
static unsigned char v_logical_not[6144];
static int64_t v_sum_1[1];
static unsigned char v_isnan[6144];
static float v_scalar_tensor[1];
static float v_where[6144];
static float v_sum_2[1];
static float v_div[1];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 6144; i++) v_alias[i] = v_args_0[i];
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_ne[i0*128 + i1*1] = (v_alias[i0*128 + i1*1] != v_alias[i0*128 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_logical_not[i0*128 + i1*1] = (!v_ne[i0*128 + i1*1]);
    }
  }
  {
    int64_t acc = 0;
    for (int r0 = 0; r0 < 48; r0++) {
      for (int r1 = 0; r1 < 128; r1++) {
        acc = (acc + v_logical_not[r0*128 + r1*1]);
      }
    }
    v_sum_1[0] = acc;
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_isnan[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] != v_args_0[i0*128 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_where[i0*128 + i1*1] = (v_isnan[i0*128 + i1*1] ? v_scalar_tensor[0] : v_args_0[i0*128 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 48; r0++) {
      for (int r1 = 0; r1 < 128; r1++) {
        acc = (acc + v_where[r0*128 + r1*1]);
      }
    }
    v_sum_2[0] = acc;
  }
  v_div[0] = (v_sum_2[0] / v_sum_1[0]);
  for (int i = 0; i < 1; i++) { out0[i] = v_div[i]; }
}
