#include <stdint.h>
#include <math.h>

static _Float16 v_view[512];
static _Float16 v_unsqueeze[512];
static _Float16 v_permute[512];
static int64_t v_arange[512];
static int64_t v_arange_1[512];
static int64_t v_unsqueeze_1[512];
static unsigned char v_eq[262144];
static unsigned char v_view_1[262144];
static _Float16 v_scalar_tensor[1];
static _Float16 v_where[262144];

extern "C" void candidate_kernel(const _Float16 *v_args_0, _Float16 *out0) {
  for (int i = 0; i < 512; i++) v_view[i] = v_args_0[i];
  for (int i = 0; i < 512; i++) v_unsqueeze[i] = v_view[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_permute[i1*1] = v_unsqueeze[i0*512 + i1];
    }
  }
  for (int i0 = 0; i0 < 512; i0++) {
    v_arange[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 512; i0++) {
    v_arange_1[i0*1] = (int64_t)(0 + i0 * 1);
  }
  for (int i = 0; i < 512; i++) v_unsqueeze_1[i] = v_arange_1[i];
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_eq[i0*512 + i1*1] = (v_arange[i1*1] == v_unsqueeze_1[i0*1]);
    }
  }
  for (int i = 0; i < 262144; i++) v_view_1[i] = v_eq[i];
  v_scalar_tensor[0] = (_Float16)(0);
  for (int i0 = 0; i0 < 512; i0++) {
    for (int i1 = 0; i1 < 512; i1++) {
      v_where[i0*512 + i1*1] = (v_view_1[i0*512 + i1*1] ? v_permute[i1*1] : v_scalar_tensor[0]);
    }
  }
  for (int i = 0; i < 262144; i++) { out0[i] = v_where[i]; }
}
