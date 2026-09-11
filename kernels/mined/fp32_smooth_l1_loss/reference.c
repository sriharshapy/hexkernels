#include <stdint.h>
#include <math.h>

static float v_sub[2048];
static float v_abs_1[2048];
static unsigned char v_lt[2048];
static float v_pow_1[2048];
static float v_mul[2048];
static float v_div[2048];
static float v_sub_1[2048];
static float v_where[2048];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] - v_args_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_abs_1[i0*64 + i1*1] = (v_sub[i0*64 + i1*1] < 0 ? -v_sub[i0*64 + i1*1] : v_sub[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_lt[i0*64 + i1*1] = (v_abs_1[i0*64 + i1*1] < 1.0f);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_pow_1[i0*64 + i1*1] = powf(v_abs_1[i0*64 + i1*1], 2);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_pow_1[i0*64 + i1*1] * 0.5f);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_div[i0*64 + i1*1] = (v_mul[i0*64 + i1*1] / 1.0f);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_1[i0*64 + i1*1] = (v_abs_1[i0*64 + i1*1] - 0.5f);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_where[i0*64 + i1*1] = (v_lt[i0*64 + i1*1] ? v_div[i0*64 + i1*1] : v_sub_1[i0*64 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 32; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_where[r0*64 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)2048);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
