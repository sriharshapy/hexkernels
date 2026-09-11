#include <stdint.h>
#include <math.h>

static float v_slice_1[504];
static float v_slice_2[504];
static float v_sub[504];
static float v_slice_3[504];
static float v_slice_4[504];
static float v_add[504];
static float v_mul[504];
static float v_sum_1[8];
static float v_div[8];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_slice_1[i0*63 + i1*1] = v_args_1[i0*64 + i1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_slice_2[i0*63 + i1*1] = v_args_1[i0*64 + (1 + i1 * 1)];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_sub[i0*63 + i1*1] = (v_slice_2[i0*63 + i1*1] - v_slice_1[i0*63 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_slice_3[i0*63 + i1*1] = v_args_0[i0*64 + i1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_slice_4[i0*63 + i1*1] = v_args_0[i0*64 + (1 + i1 * 1)];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_add[i0*63 + i1*1] = (v_slice_3[i0*63 + i1*1] + v_slice_4[i0*63 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 63; i1++) {
      v_mul[i0*63 + i1*1] = (v_add[i0*63 + i1*1] * v_sub[i0*63 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 63; r1++) {
        acc = (acc + v_mul[i0*63 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_div[i0*1] = (v_sum_1[i0*1] / 2.0f);
  }
  for (int i = 0; i < 8; i++) { out0[i] = v_div[i]; }
}
