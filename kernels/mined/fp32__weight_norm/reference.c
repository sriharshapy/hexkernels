#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[256];
static float v_div[98304];
static float v_mul[98304];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 384; r1++) {
        acc = (acc + v_args_0[i0*384 + r1*1]*v_args_0[i0*384 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_div[i0*384 + i1*1] = (v_args_1[i0*384 + i1*1] / v_linalg_vector_norm[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] * v_div[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_mul[i]; }
}
