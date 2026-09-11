#include <stdint.h>
#include <math.h>

static float v_pow_1[512];
static float v_sum_1[8];
static float v_full_like[8];
static float v_pow_2[512];
static float v_sum_2[8];
static float v_full_like_1[8];
static float v_mul[512];
static float v_cat[528];
static float v_cat_1[528];
static float v_permute[528];
static float v_mm[64];
static float v_clamp[64];
static float v_sqrt[64];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_pow_1[i0*64 + i1*1] = powf(v_args_0[i0*64 + i1*1], 2);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_pow_1[i0*64 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_full_like[i0*1] = (float)(1);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_pow_2[i0*64 + i1*1] = powf(v_args_1[i0*64 + i1*1], 2);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_pow_2[i0*64 + r1*1]);
      }
      v_sum_2[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_full_like_1[i0*1] = (float)(1);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] * -2);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_cat[(i0)*66 + (i1)] = v_mul[i0*64 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat[(i0)*66 + (i1 + 64)] = v_sum_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat[(i0)*66 + (i1 + 65)] = v_full_like[i0*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_cat_1[(i0)*66 + (i1)] = v_args_1[i0*64 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_1[(i0)*66 + (i1 + 64)] = v_full_like_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_cat_1[(i0)*66 + (i1 + 65)] = v_sum_2[i0*1];
    }
  }
  for (int i0 = 0; i0 < 66; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_permute[i0*8 + i1*1] = v_cat_1[i1*66 + i0];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      float acc = 0;
      for (int k = 0; k < 66; k++) {
        acc = acc + v_cat[i0*66 + k] * v_permute[k*8 + i1];
      }
    v_mm[i0*8 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_clamp[i0*8 + i1*1] = (v_mm[i0*8 + i1*1] < 0 ? 0 : v_mm[i0*8 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_sqrt[i0*8 + i1*1] = sqrtf(v_clamp[i0*8 + i1*1]);
    }
  }
  for (int i = 0; i < 64; i++) { out0[i] = v_sqrt[i]; }
}
