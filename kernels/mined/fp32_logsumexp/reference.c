#include <stdint.h>
#include <math.h>

static float v_amax[32];
static float v_abs_1[32];
static unsigned char v_eq[32];
static float v_scalar_tensor[1];
static float v_where[32];
static float v_squeeze[32];
static float v_sub[32];
static float v_exp[32];
static float v_sum_1[32];
static float v_log[32];
static float v_add[32];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 4; i1++) {
    for (int i2 = 0; i2 < 8; i2++) {
      {
        float acc = -INFINITY;
        for (int r0 = 0; r0 < 1; r0++) {
          acc = (v_args_0[i1*8 + i2*1] > acc ? v_args_0[i1*8 + i2*1] : acc);
        }
        v_amax[i1*8 + i2*1] = acc;
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_abs_1[i1*8 + i2*1] = (v_amax[i1*8 + i2*1] < 0 ? -v_amax[i1*8 + i2*1] : v_amax[i1*8 + i2*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_eq[i1*8 + i2*1] = (v_abs_1[i1*8 + i2*1] == INFINITY);
      }
    }
  }
  v_scalar_tensor[0] = (float)(0.0f);
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_where[i1*8 + i2*1] = (v_eq[i1*8 + i2*1] ? v_scalar_tensor[0] : v_amax[i1*8 + i2*1]);
      }
    }
  }
  for (int i = 0; i < 32; i++) v_squeeze[i] = v_where[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_sub[i1*8 + i2*1] = (v_args_0[i1*8 + i2*1] - v_where[i1*8 + i2*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_exp[i1*8 + i2*1] = expf(v_sub[i1*8 + i2*1]);
      }
    }
  }
  for (int i1 = 0; i1 < 4; i1++) {
    for (int i2 = 0; i2 < 8; i2++) {
      {
        float acc = 0;
        for (int r0 = 0; r0 < 1; r0++) {
          acc = (acc + v_exp[i1*8 + i2*1]);
        }
        v_sum_1[i1*8 + i2*1] = acc;
      }
    }
  }
  for (int i0 = 0; i0 < 4; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_log[i0*8 + i1*1] = logf(v_sum_1[i0*8 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 4; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_add[i0*8 + i1*1] = (v_log[i0*8 + i1*1] + v_squeeze[i0*8 + i1*1]);
    }
  }
  for (int i = 0; i < 32; i++) { out0[i] = v_add[i]; }
}
