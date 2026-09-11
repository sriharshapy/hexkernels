#include <stdint.h>
#include <math.h>

static unsigned char v_gt[6144];

extern "C" void candidate_kernel(const float *v_args_0, unsigned char *out0) {
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_gt[i0*128 + i1*1] = (v_args_0[i0*128 + i1*1] > 2.0f);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_gt[i]; }
}
