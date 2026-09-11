#include <stdint.h>
#include <math.h>

static int64_t v_arange[128];
static int64_t v_unsqueeze[128];
static int64_t v_arange_1[48];
static int64_t v_unsqueeze_1[48];
static int64_t v_sub[6144];
static unsigned char v_ge[6144];
static float v_scalar_tensor[1];
static float v_where[6144];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 128; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i = 0; i < 128; i++) v_unsqueeze[i] = v_arange[i];
  for (int i0 = 0; i0 < 48; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i = 0; i < 48; i++) v_unsqueeze_1[i] = v_arange_1[i];
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_sub[i0*128 + i1*1] = (v_unsqueeze[i1*1] - v_unsqueeze_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_ge[i0*128 + i1*1] = (v_sub[i0*128 + i1*1] >= 0);
    }
  }
  v_scalar_tensor[0] = (float)(0);
  for (int i0 = 0; i0 < 48; i0++) {
    for (int i1 = 0; i1 < 128; i1++) {
      v_where[i0*128 + i1*1] = (v_ge[i0*128 + i1*1] ? v_args_0[i0*128 + i1*1] : v_scalar_tensor[0]);
    }
  }
  for (int i = 0; i < 6144; i++) { out0[i] = v_where[i]; }
}
