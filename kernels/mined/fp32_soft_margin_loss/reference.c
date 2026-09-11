#include <stdint.h>
#include <math.h>

static float v_neg[2048];
static float v_mul[2048];
static float v_exp[2048];
static float v_log1p[2048];
static float v_mean[1];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_neg[i0*64 + i1*1] = (-v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_neg[i0*64 + i1*1] * v_args_1[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_exp[i0*64 + i1*1] = expf(v_mul[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_log1p[i0*64 + i1*1] = log1pf(v_exp[i0*64 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 32; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_log1p[r0*64 + r1*1]);
      }
    }
    v_mean[0] = (acc / (float)2048);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_mean[i]; }
}
