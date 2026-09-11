#include <stdint.h>
#include <math.h>

static unsigned char v_gt[1536];
static float v_mul[1536];
static float v_where[1536];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int i0 = 0; i0 < 24; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_gt[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] > 0);
    }
  }
  for (int i0 = 0; i0 < 24; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_mul[i0*64 + i1*1] = (v_args_1[i0*64 + i1*1] * v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 24; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_where[i0*64 + i1*1] = (v_gt[i0*64 + i1*1] ? v_args_0[i0*64 + i1*1] : v_mul[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 1536; i++) { out0[i] = v_where[i]; }
}
