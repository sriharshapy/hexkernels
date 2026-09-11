#include <stdint.h>
#include <math.h>

static float v__to_copy[524288];
static float v_arange[16];
static float v_add[16];
static float v_mul[16];
static int64_t v__to_copy_1[16];
static int64_t v_unsqueeze[16];
static int64_t v_unsqueeze_1[16];
static float v_arange_1[16];
static float v_add_1[16];
static float v_mul_1[16];
static int64_t v__to_copy_2[16];
static int64_t v_unsqueeze_2[16];
static float v_arange_2[16];
static float v_add_2[16];
static float v_mul_2[16];
static int64_t v__to_copy_3[16];
static float v_index[65536];
static float v__to_copy_4[65536];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 32; i2++) {
        for (int i3 = 0; i3 < 32; i3++) {
          for (int i4 = 0; i4 < 32; i4++) {
            v__to_copy[i1*32768 + i2*1024 + i3*32 + i4*1] = v_args_0[i1*32768 + i2*1024 + i3*32 + i4*1];
          }
        }
      }
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add[i0*1] = (v_arange[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_1[i0*1] = v_mul[i0*1];
  }
  for (int i = 0; i < 16; i++) v_unsqueeze[i] = v__to_copy_1[i];
  for (int i = 0; i < 16; i++) v_unsqueeze_1[i] = v_unsqueeze[i];
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange_1[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add_1[i0*1] = (v_arange_1[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul_1[i0*1] = (v_add_1[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_2[i0*1] = v_mul_1[i0*1];
  }
  for (int i = 0; i < 16; i++) v_unsqueeze_2[i] = v__to_copy_2[i];
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange_2[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add_2[i0*1] = (v_arange_2[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul_2[i0*1] = (v_add_2[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_3[i0*1] = v_mul_2[i0*1];
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            v_index[i1*4096 + i2*256 + i3*16 + i4*1] = v__to_copy[(i0)*524288 + (i1)*32768 + (v_unsqueeze_1[(i2)])*1024 + (v_unsqueeze_2[(i3)])*32 + (v__to_copy_3[(i4)])];
          }
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            v__to_copy_4[i1*4096 + i2*256 + i3*16 + i4*1] = v_index[i1*4096 + i2*256 + i3*16 + i4*1];
          }
        }
      }
    }
  }
  for (int i = 0; i < 65536; i++) { out0[i] = v__to_copy_4[i]; }
}
