#include <stdint.h>
#include <math.h>

static int64_t v_arange[1280];
static int64_t v_add[1280];
static int64_t v_fmod[1280];
static float v_index_select[1310720];
static int64_t v_arange_1[1024];
static int64_t v_add_1[1024];
static int64_t v_fmod_1[1024];
static float v_index_select_1[1310720];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1280; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    v_add[i0*1] = (v_arange[i0*1] + 640);
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    v_fmod[i0*1] = fmodf(v_add[i0*1], 1280);
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_index_select[i0*1024 + i1*1] = v_args_0[(v_fmod[i0])*1024 + (i1)];
    }
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_add_1[i0*1] = (v_arange_1[i0*1] + 512);
  }
  for (int i0 = 0; i0 < 1024; i0++) {
    v_fmod_1[i0*1] = fmodf(v_add_1[i0*1], 1024);
  }
  for (int i0 = 0; i0 < 1280; i0++) {
    for (int i1 = 0; i1 < 1024; i1++) {
      v_index_select_1[i0*1024 + i1*1] = v_index_select[(i0)*1024 + (v_fmod_1[i1])];
    }
  }
  for (int i = 0; i < 1310720; i++) { out0[i] = v_index_select_1[i]; }
}
