#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[384];
static float v_div[196608];
static float v_mul[196608];
static float v_amin[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, float *out0) {
  for (int i0 = 0; i0 < 384; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_xs_0[i0*512 + r1*1]*v_xs_0[i0*512 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_div[i0*512 + i1*1] = (v_xs_1[i0*512 + i1*1] / v_linalg_vector_norm[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_mul[i0*512 + i1*1] = (v_xs_0[i0*512 + i1*1] * v_div[i0*512 + i1*1]);
    }
  }
  {
    float acc = INFINITY;
    for (int r0 = 0; r0 < 384; r0++) {
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (v_mul[r0*512 + r1*1] < acc ? v_mul[r0*512 + r1*1] : acc);
      }
    }
    v_amin[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_amin[i]; }
}
