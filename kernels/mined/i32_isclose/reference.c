#include <stdint.h>
#include <math.h>

static unsigned char v_eq[98304];
static float v__to_copy[98304];
static float v_mul[98304];
static float v_abs_1[98304];
static float v_add[98304];
static float v_sub[98304];
static float v_abs_2[98304];
static float v_abs_3[98304];
static unsigned char v_ne[98304];
static unsigned char v_eq_1[98304];
static unsigned char v_mul_1[98304];
static unsigned char v_le[98304];
static unsigned char v_bitwise_and[98304];
static unsigned char v_bitwise_or[98304];

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] == v_args_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v__to_copy[i0*384 + i1*1] = v_args_1[i0*384 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v__to_copy[i0*384 + i1*1] * 1e-05f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_1[i0*384 + i1*1] = (v_mul[i0*384 + i1*1] < 0 ? -v_mul[i0*384 + i1*1] : v_mul[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_add[i0*384 + i1*1] = (v_abs_1[i0*384 + i1*1] + 1e-08f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_sub[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] - v__to_copy[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_2[i0*384 + i1*1] = (v_sub[i0*384 + i1*1] < 0 ? -v_sub[i0*384 + i1*1] : v_sub[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_3[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] < 0 ? -v_abs_2[i0*384 + i1*1] : v_abs_2[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_ne[i0*384 + i1*1] = (v_abs_3[i0*384 + i1*1] != INFINITY);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq_1[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] == v_abs_2[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul_1[i0*384 + i1*1] = (v_eq_1[i0*384 + i1*1] * v_ne[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_le[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] <= v_add[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_bitwise_and[i0*384 + i1*1] = (v_mul_1[i0*384 + i1*1] & v_le[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_bitwise_or[i0*384 + i1*1] = (v_eq[i0*384 + i1*1] | v_bitwise_and[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_bitwise_or[i]; }
}
