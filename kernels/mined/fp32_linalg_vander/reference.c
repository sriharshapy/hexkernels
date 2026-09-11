#include <stdint.h>
#include <math.h>

static float v_unsqueeze[6144];
static float v_expand[780288];
static float v_cumprod[780288];
static float v_full[6144];
static float v_cat[786432];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 6144; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 127; i2++) {
        v_expand[i0*16256 + i1*127 + i2*1] = v_unsqueeze[i0*128 + i1];
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      float acc = 1;
      for (int s2 = 0; s2 < 127; s2++) {
        acc = (acc * v_expand[i0*16256 + i1*127 + s2*1]);
        v_cumprod[i0*16256 + i1*127 + s2*1] = (float)acc;
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_full[i0*128 + i1*1] = (float)(1);
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 1; i2++) {
        v_cat[(i0)*16384 + (i1)*128 + (i2)] = v_full[i0*128 + i1*1];
      }
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      for (int i2 = 0; i2 < 127; i2++) {
        v_cat[(i0)*16384 + (i1)*128 + (i2 + 1)] = v_cumprod[i0*16256 + i1*127 + i2*1];
      }
    }
  }
  for (int i = 0; i < 786432; i++) { out0[i] = v_cat[i]; }
}
