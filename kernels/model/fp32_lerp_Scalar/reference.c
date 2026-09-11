#include <stdint.h>
#include <math.h>

static float v_full[1];
static float v_abs_1[1];
static unsigned char v_ge[1];
static float v_sub[1];
static float v_where[1];
static float v_where_1[512];
static float v_sub_1[512];
static float v_mul[512];
static float v_add[512];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  v_full[0] = (float)(2.0f);
  v_abs_1[0] = (v_full[0] < 0 ? -v_full[0] : v_full[0]);
  v_ge[0] = (v_abs_1[0] >= 0.5f);
  v_sub[0] = (v_full[0] - 1);
  v_where[0] = (v_ge[0] ? v_sub[0] : v_full[0]);
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_where_1[i0*64 + i1*1] = (v_ge[0] ? v_args_1[i0*64 + i1*1] : v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_1[i0*64 + i1*1] = (v_args_1[i0*64 + i1*1] - v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_where[0] * v_sub_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_add[i0*64 + i1*1] = (v_mul[i0*64 + i1*1] + v_where_1[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_add[i]; }
}
