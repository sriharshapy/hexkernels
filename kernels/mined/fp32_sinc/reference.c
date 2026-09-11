#include <stdint.h>
#include <math.h>

static float v_mul[1310720];
static unsigned char v_eq[1310720];
static float v_sin[1310720];
static float v_div[1310720];
static float v_scalar_tensor[1];
static float v_where[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_mul[i0*1024 + i1*1] = (v_args_0[i0*1024 + i1*1] * 3.141592653589793f);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_eq[i0*1024 + i1*1] = (v_mul[i0*1024 + i1*1] == 0);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_sin[i0*1024 + i1*1] = sinf(v_mul[i0*1024 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_div[i0*1024 + i1*1] = (v_sin[i0*1024 + i1*1] / v_mul[i0*1024 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (float)(1);
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_where[i0*1024 + i1*1] = (v_eq[i0*1024 + i1*1] ? v_scalar_tensor[0] : v_div[i0*1024 + i1*1]);
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_where[i]; }
}
