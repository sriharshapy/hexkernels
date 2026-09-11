#include <stdint.h>
#include <math.h>

static float v_mul[98304];
static float v_permute[98304];
static float v_mul_1[98304];
static float v_mm[65536];
static float v_amax[256];
static float v_sub[65536];
static float v_exp[65536];
static float v_sum_1[256];
static float v_div[65536];
static unsigned char v_eq[65536];
static unsigned char v_logical_not[65536];
static unsigned char v_any_1[256];
static unsigned char v_logical_not_1[256];
static float v_full_like[65536];
static float v_where[65536];
static float v_mm_1[98304];
static float v__to_copy_1[98304];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, const float *v_args_2, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] * 0.22590050090246122f);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_permute[i0*256 + i1*1] = v_args_1[i1*384 + i0];
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_mul_1[i0*256 + i1*1] = (v_permute[i0*256 + i1*1] * 0.22590050090246122f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      float acc = 0;
      for (int k = 0; k < 384; k++) {
        acc = acc + v_mul[i0*384 + k] * v_mul_1[k*256 + i1];
      }
    v_mm[i0*256 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      float acc = -INFINITY;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (v_mm[i0*256 + r1*1] > acc ? v_mm[i0*256 + r1*1] : acc);
      }
      v_amax[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_sub[i0*256 + i1*1] = (v_mm[i0*256 + i1*1] - v_amax[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_exp[i0*256 + i1*1] = expf(v_sub[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (acc + v_exp[i0*256 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_div[i0*256 + i1*1] = (v_exp[i0*256 + i1*1] / v_sum_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_eq[i0*256 + i1*1] = (v_mm[i0*256 + i1*1] == (-INFINITY));
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_logical_not[i0*256 + i1*1] = (!v_eq[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      int32_t acc = 0;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (acc || (v_logical_not[i0*256 + r1*1] != 0));
      }
      v_any_1[i0*1] = (unsigned char)(acc);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_logical_not_1[i0*1] = (!v_any_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_full_like[i0*256 + i1*1] = (float)(0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_where[i0*256 + i1*1] = (v_logical_not_1[i0*1] ? v_full_like[i0*256 + i1*1] : v_div[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      float acc = 0;
      for (int k = 0; k < 256; k++) {
        acc = acc + v_where[i0*256 + k] * v_args_2[k*384 + i1];
      }
    v_mm_1[i0*384 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v__to_copy_1[i0*384 + i1*1] = v_mm_1[i0*384 + i1*1];
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v__to_copy_1[i]; }
}
