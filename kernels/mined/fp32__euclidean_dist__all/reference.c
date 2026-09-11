#include <stdint.h>
#include <math.h>

static float v_pow_1[1048576];
static float v_sum_1[1024];
static float v_full_like[1024];
static float v_pow_2[1048576];
static float v_sum_2[1024];
static float v_full_like_1[1024];
static float v_mul[1048576];
static float v_cat[1050624];
static float v_cat_1[1050624];
static float v_permute[1050624];
static float v_mm[1048576];
static float v_clamp[1048576];
static float v_sqrt[1048576];
static unsigned char v_logical_not[1048576];
static unsigned char v_any_1[1];
static unsigned char v_logical_not_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_pow_1[i0*1024 + i1*1] = powf(v_xs_0[i0*1024 + i1*1], 2);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc + v_pow_1[i0*1024 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_full_like[i0*1] = (float)(1);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_pow_2[i0*1024 + i1*1] = powf(v_xs_1[i0*1024 + i1*1], 2);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc + v_pow_2[i0*1024 + r1*1]);
      }
      v_sum_2[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_full_like_1[i0*1] = (float)(1);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_xs_0[i0*1024 + i1*1] * -2);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_cat[(i0)*1026 + (i1)] = v_mul[i0*1024 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat[(i0)*1026 + (i1 + 1024)] = v_sum_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat[(i0)*1026 + (i1 + 1025)] = v_full_like[i0*1];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_cat_1[(i0)*1026 + (i1)] = v_xs_1[i0*1024 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_1[(i0)*1026 + (i1 + 1024)] = v_full_like_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_1[(i0)*1026 + (i1 + 1025)] = v_sum_2[i0*1];
    }
  }
  for (int i0 = 0; i0 < 1026; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_permute[i0*1024 + i1*1] = v_cat_1[i1*1026 + i0];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      float acc = 0;
      for (int k = 0; k < 1026; k++) {
        acc = acc + v_cat[i0*1026 + k] * v_permute[k*1024 + i1];
      }
    v_mm[i0*1024 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_clamp[i0*1024 + i1*1] = (v_mm[i0*1024 + i1*1] < 0 ? 0 : v_mm[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sqrt[i0*1024 + i1*1] = sqrtf(v_clamp[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_logical_not[i0*1024 + i1*1] = (!v_sqrt[i0*1024 + i1*1]);
    }
  }
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 1024; r0++) {
      for (int r1 = 0; r1 < 1024; r1++) {
        acc = (acc || (v_logical_not[r0*1024 + r1*1] != 0));
      }
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  v_logical_not_1[0] = (!v_any_1[0]);
  for (int i = 0; i < 1; i++) { out0[i] = v_logical_not_1[i]; }
}
