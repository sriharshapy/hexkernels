#include <stdint.h>
#include <math.h>

static float v_amax[512];
static float v_sub[196608];
static float v_exp[196608];
static float v_sum_1[512];
static float v_log[512];
static float v_sub_1[196608];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 512; i1++) {
    {
      float acc = -INFINITY;
      for (int r0 = 0; r0 < 384; r0++) {
        acc = (v_args_0[r0*512 + i1*1] > acc ? v_args_0[r0*512 + i1*1] : acc);
      }
      v_amax[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] - v_amax[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_exp[i0*512 + i1*1] = expf(v_sub[i0*512 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 512; i1++) {
    {
      float acc = 0;
      for (int r0 = 0; r0 < 384; r0++) {
        acc = (acc + v_exp[r0*512 + i1*1]);
      }
      v_sum_1[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_log[i1*1] = logf(v_sum_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub_1[i0*512 + i1*1] = (v_sub[i0*512 + i1*1] - v_log[i1*1]);
    }
  }
  for (int i = 0; i < 196608; i++) { out0[i] = v_sub_1[i]; }
}
