#include <stdint.h>
#include <math.h>

static float v_sub[196608];
static float v_add[196608];
static float v_linalg_vector_norm[384];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_sub[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] - v_args_1[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_add[i0*512 + i1*1] = (v_sub[i0*512 + i1*1] + 1e-06f);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_add[i0*512 + r1*1]*v_add[i0*512 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i = 0; i < 384; i++) { out0[i] = v_linalg_vector_norm[i]; }
}
