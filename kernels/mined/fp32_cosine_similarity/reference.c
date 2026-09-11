#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[384];
static float v_clone[384];
static float v_linalg_vector_norm_1[384];
static float v_clone_1[384];
static float v_clamp[384];
static float v_clamp_1[384];
static float v_div[196608];
static float v_div_1[196608];
static float v_mul[196608];
static float v_sum_1[384];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 384; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_args_0[i0*512 + r1*1]*v_args_0[i0*512 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clone[i0*1] = v_linalg_vector_norm[i0*1];
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_args_1[i0*512 + r1*1]*v_args_1[i0*512 + r1*1]);
      }
      v_linalg_vector_norm_1[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clone_1[i0*1] = v_linalg_vector_norm_1[i0*1];
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp[i0*1] = (v_clone[i0*1] < 1e-08f ? 1e-08f : v_clone[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_clamp_1[i0*1] = (v_clone_1[i0*1] < 1e-08f ? 1e-08f : v_clone_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_div[i0*512 + i1*1] = (v_args_1[i0*512 + i1*1] / v_clamp_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_div_1[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] / v_clamp[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_mul[i0*512 + i1*1] = (v_div_1[i0*512 + i1*1] * v_div[i0*512 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_mul[i0*512 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i = 0; i < 384; i++) { out0[i] = v_sum_1[i]; }
}
