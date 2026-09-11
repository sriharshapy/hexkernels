#include <stdint.h>
#include <math.h>

static float v__to_copy[4096];
static float v_arange[8];
static float v_add[8];
static float v_mul[8];
static int64_t v__to_copy_1[8];
static int64_t v_unsqueeze[8];
static float v_arange_1[8];
static float v_add_1[8];
static float v_mul_1[8];
static int64_t v__to_copy_2[8];
static float v_index[1024];
static float v__to_copy_3[1024];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          v__to_copy[i1*256 + i2*16 + i3*1] = v_args_0[i1*256 + i2*16 + i3*1];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_arange[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_add[i0*1] = (v_arange[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_1[i0*1] = v_mul[i0*1];
  }
  for (int i = 0; i < 8; i++) v_unsqueeze[i] = v__to_copy_1[i];
  for (int i0 = 0; i0 < 8; i0++) {
    v_arange_1[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_add_1[i0*1] = (v_arange_1[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v_mul_1[i0*1] = (v_add_1[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 8; i0++) {
    v__to_copy_2[i0*1] = v_mul_1[i0*1];
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v_index[i1*64 + i2*8 + i3*1] = v__to_copy[(i0)*4096 + (i1)*256 + (v_unsqueeze[(i2)])*16 + (v__to_copy_2[(i3)])];
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          v__to_copy_3[i1*64 + i2*8 + i3*1] = v_index[i1*64 + i2*8 + i3*1];
        }
      }
    }
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v__to_copy_3[i]; }
}
