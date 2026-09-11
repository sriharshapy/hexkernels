#include <stdint.h>
#include <math.h>

static float v_abs_1[512];
static unsigned char v_le[512];
static float v_scalar_tensor[1];
static float v__to_copy[512];
static float v_where[512];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_abs_1[i0*64 + i1*1] = (v_args_0[i0*64 + i1*1] < 0 ? -v_args_0[i0*64 + i1*1] : v_args_0[i0*64 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_le[i0*64 + i1*1] = (v_abs_1[i0*64 + i1*1] <= 0.5f);
    }
  }
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v__to_copy[i0*64 + i1*1] = v_args_0[i0*64 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_where[i0*64 + i1*1] = (v_le[i0*64 + i1*1] ? v_scalar_tensor[0] : v__to_copy[i0*64 + i1*1]);
    }
  }
  for (int i = 0; i < 512; i++) { out0[i] = v_where[i]; }
}
