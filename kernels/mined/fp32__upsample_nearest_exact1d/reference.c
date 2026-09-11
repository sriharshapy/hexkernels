#include <stdint.h>
#include <math.h>

static float v__to_copy[4096];
static float v_arange[32];
static float v_add[32];
static float v_mul[32];
static int64_t v__to_copy_1[32];
static float v_index[2048];
static float v__to_copy_2[2048];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 64; i2++) {
        v__to_copy[i1*64 + i2*1] = v_args_0[i1*64 + i2*1];
      }
    }
  }
  for (int i0 = 0; i0 < 32; i0++) {
    v_arange[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 32; i0++) {
    v_add[i0*1] = (v_arange[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 32; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 32; i0++) {
    v__to_copy_1[i0*1] = v_mul[i0*1];
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 32; i2++) {
        v_index[i1*32 + i2*1] = v__to_copy[(i0)*4096 + (i1)*64 + (v__to_copy_1[(i2)])];
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      for (int i2 = 0; i2 < 32; i2++) {
        v__to_copy_2[i1*32 + i2*1] = v_index[i1*32 + i2*1];
      }
    }
  }
  for (int i = 0; i < 2048; i++) { out0[i] = v__to_copy_2[i]; }
}
