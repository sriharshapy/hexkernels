#include <stdint.h>
#include <math.h>

static float v_unsqueeze[32768];
static float v_permute[32768];
static int64_t v_arange[256];
static int64_t v_arange_1[256];
static int64_t v_unsqueeze_1[256];
static unsigned char v_eq[65536];
static unsigned char v_view[65536];
static float v_scalar_tensor[1];
static float v_where[8388608];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i = 0; i < 32768; i++) v_unsqueeze[i] = v_args_0[i];
  for (int i0 = 0; i0 < 128; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      for (int i2 = 0; i2 < 256; i2++) {
        v_permute[i0*256 + i2*1] = v_unsqueeze[i0*256 + i1*256 + i2];
      }
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 256; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i = 0; i < 256; i++) v_unsqueeze_1[i] = v_arange_1[i];
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_eq[i0*256 + i1*1] = (v_arange[i1*1] == v_unsqueeze_1[i0*1]);
    }
  }
  for (int i = 0; i < 65536; i++) v_view[i] = v_eq[i];
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 128; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      for (int i2 = 0; i2 < 256; i2++) {
        v_where[i0*65536 + i1*256 + i2*1] = (v_view[i1*256 + i2*1] ? v_permute[i0*256 + i2*1] : v_scalar_tensor[0]);
      }
    }
  }
  for (int i = 0; i < 8388608; i++) { out0[i] = v_where[i]; }
}
