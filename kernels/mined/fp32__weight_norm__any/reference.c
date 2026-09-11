#include <stdint.h>
#include <math.h>

static float v_linalg_vector_norm[32];
static float v_div[2048];
static float v_mul[2048];
static unsigned char v_any_1[1];

extern "C" void candidate_kernel(const float *v_xs_0, const float *v_xs_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 32; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc + v_xs_0[i0*64 + r1*1]*v_xs_0[i0*64 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_div[i0*64 + i1*1] = (v_xs_1[i0*64 + i1*1] / v_linalg_vector_norm[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_xs_0[i0*64 + i1*1] * v_div[i0*64 + i1*1]);
    }
  }
  {
    int32_t acc = 0;
    for (int r0 = 0; r0 < 32; r0++) {
      for (int r1 = 0; r1 < 64; r1++) {
        acc = (acc || (v_mul[r0*64 + r1*1] != 0));
      }
    }
    v_any_1[0] = (unsigned char)(acc);
  }
  for (int i = 0; i < 1; i++) { out0[i] = v_any_1[i]; }
}
