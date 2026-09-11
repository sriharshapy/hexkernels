#include <stdint.h>
#include <math.h>

static unsigned char v_isnan[262144];
static float v_scalar_tensor[1];
static float v_where[262144];
static float v_sum_1[1];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_isnan[i0*512 + i1*1] = (v_args_0[i0*512 + i1*1] != v_args_0[i0*512 + i1*1]);
    }
  }
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_where[i0*512 + i1*1] = (v_isnan[i0*512 + i1*1] ? v_scalar_tensor[0] : v_args_0[i0*512 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 512; r0++) {
      for (int r1 = 0; r1 < 512; r1++) {
        acc = (acc + v_where[r0*512 + r1*1]);
      }
    }
    v_sum_1[0] = acc;
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_sum_1[i]; }
}
