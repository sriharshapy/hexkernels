#include <stdint.h>
#include <math.h>

static float v_exp[512];
static float v_linalg_vector_norm[1];

extern "C" void candidate_kernel(const float *v_xs_0, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_exp[i0*64 + i1*1] = expf(v_xs_0[i0*64 + i1*1]);
    }
  }
  {
    float acc = 0;
    for (int r0 = 0; r0 < 8; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_exp[r0*64 + r1*1]*v_exp[r0*64 + r1*1]);
      }
    }
    v_linalg_vector_norm[0] = sqrtf(acc);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_linalg_vector_norm[i]; }
}
