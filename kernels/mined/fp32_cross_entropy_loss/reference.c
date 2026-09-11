#include <stdint.h>
#include <math.h>

static float v__to_copy[512];
static float v_amax[8];
static float v_sub[512];
static float v_exp[512];
static float v_sum_1[8];
static float v_log[8];
static float v_sub_1[512];
static float v_mul[512];
static float v_sum_2[1];
static float v_neg[1];
static float v_div[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v__to_copy[i0*64 + i1*1] = v_args_0[i0*64 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = -INFINITY;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (v__to_copy[i0*64 + r1*1] > acc ? v__to_copy[i0*64 + r1*1] : acc);
      }
      v_amax[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub[i0*64 + i1*1] = (v__to_copy[i0*64 + i1*1] - v_amax[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_exp[i0*64 + i1*1] = expf(v_sub[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_exp[i0*64 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_log[i0*1] = logf(v_sum_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_sub_1[i0*64 + i1*1] = (v_sub[i0*64 + i1*1] - v_log[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_sub_1[i0*64 + i1*1] * v_args_1[i0*64 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_mul[r0*64 + r1*1]);
      }
    }
    v_sum_2[0] = acc;
  }
  v_neg[0] = (-v_sum_2[0]);
  v_div[0] = (v_neg[0] / 8);
  for (int i = 0; i < 1; i++) { out0[i] = v_div[i]; }
}
