#include <stdint.h>
#include <math.h>

static float v_amax[1024];
static float v_sub[1310720];
static float v_exp[1310720];
static float v_sum_1[1024];
static float v_div[1310720];
static unsigned char v_eq[1310720];
static unsigned char v_logical_not[1310720];
static unsigned char v_any_1[1024];
static unsigned char v_logical_not_1[1024];
static float v_full_like[1310720];
static float v_where[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      float acc = -INFINITY;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (v_args_0[r0*1024 + i1*1] > acc ? v_args_0[r0*1024 + i1*1] : acc);
      }
      v_amax[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sub[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] - v_amax[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_exp[i0*1024 + i1*1] = expf(v_sub[i0*1024 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      float acc = 0;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (acc + v_exp[r0*1024 + i1*1]);
      }
      v_sum_1[i1*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_div[i0*1024 + i1*1] = (v_exp[i0*1024 + i1*1] / v_sum_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_eq[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] == (-INFINITY));
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_not[i0*1024 + i1*1] = (!v_eq[i0*1024 + i1*1]);
    }
  }
  for (int i1 = 0; i1 < 1024; i1++) {
    {
      int32_t acc = 0;
      for (int r0 = 0; r0 < 1280; r0++) {
        acc = (acc || (v_logical_not[r0*1024 + i1*1] != 0));
      }
      v_any_1[i1*1] = (unsigned char)(acc);
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_not_1[i1*1] = (!v_any_1[i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_full_like[i0*1024 + i1*1] = (float)(0);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_logical_not_1[i1*1] ? v_full_like[i0*1024 + i1*1] : v_div[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_where[i]; }
}
