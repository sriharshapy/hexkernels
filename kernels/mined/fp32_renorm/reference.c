#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[8];
static unsigned char v_gt[8];
static float v_add[8];
static float v_reciprocal[8];
static float v_mul[8];
static float v_scalar_tensor[1];
static float v_where[8];
static float v_mul_1[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_args_0[i0*64 + r1*1]*v_args_0[i0*64 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_gt[i0*1] = (v_linalg_vector_norm[i0*1] > 2.0f);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_add[i0*1] = (v_linalg_vector_norm[i0*1] + 1e-07f);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_reciprocal[i0*1] = (1.0f / v_add[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_mul[i0*1] = (v_reciprocal[i0*1] * 2.0f);
    }
  }
  v_scalar_tensor[0] = (float)(1.0f);
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_where[i0*1] = (v_gt[i0*1] ? v_mul[i0*1] : v_scalar_tensor[0]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul_1[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] * v_where[i0*1]);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_mul_1[i]; }
}
