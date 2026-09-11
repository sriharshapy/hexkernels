#include <stdint.h>
#include <math.h>

static float v_permute[1024];
static float v_mm[256];
static float v_permute_1[1024];
static float v_mm_1[256];
static float v_add[256];
static float v_relu[256];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, const float *v_args_2, const float *v_args_3, float *out0) {
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      v_permute[i0*16 + i1*1] = v_args_3[i1*64 + i0];
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      float acc = 0;
      for (int k = 0; k < 64; k++) {
        acc = acc + v_args_1[i0*64 + k] * v_permute[k*16 + i1];
      }
    v_mm[i0*16 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      v_permute_1[i0*16 + i1*1] = v_args_2[i1*64 + i0];
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      float acc = 0;
      for (int k = 0; k < 64; k++) {
        acc = acc + v_args_0[i0*64 + k] * v_permute_1[k*16 + i1];
      }
    v_mm_1[i0*16 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      v_add[i0*16 + i1*1] = (v_mm[i0*16 + i1*1] + v_mm_1[i0*16 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      v_relu[i0*16 + i1*1] = (v_add[i0*16 + i1*1] > 0 ? v_add[i0*16 + i1*1] : 0);
    }
  }
  for (int i = 0; i < 256; i++) { out0[i] = v_relu[i]; }
}
