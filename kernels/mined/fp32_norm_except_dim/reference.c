#include <stdint.h>
#include <math.h>

static float v_view[6144];
static float v_linalg_vector_norm[48];
static float v_view_1[48];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 6144; i++) v_view[i] = v_args_0[i];
  for (int i0 = 0; i0 < 48; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 128; r1++) {
        acc = (acc + v_view[i0*128 + r1*1]*v_view[i0*128 + r1*1]);
      }
      v_linalg_vector_norm[i0*1] = sqrtf(acc);
    }
  }
  for (int i = 0; i < 48; i++) v_view_1[i] = v_linalg_vector_norm[i];
  for (int i = 0; i < 48; i++) { out0[i] = v_view_1[i]; }
}
