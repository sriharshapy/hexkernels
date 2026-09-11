#include <stdint.h>
#include <math.h>

static _Float16 v_permute[2048];
static _Float16 v_view[2048];
static _Float16 v_permute_1[2048];
static _Float16 v_view_1[2048];
static _Float16 v_mm[1024];
static _Float16 v_view_2[1024];

extern "C" void candidate_kernel(const _Float16 *v_args_0, const _Float16 *v_args_1, _Float16 *out0) {
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 64; i1++) {
      v_permute[i0*64 + i1*1] = v_args_0[i0*64 + i1];
    }
  }
  for (int i = 0; i < 2048; i++) v_view[i] = v_permute[i];
  for (int i0 = 0; i0 < 64; i0++) {
    for (int i1 = 0; i1 < 32; i1++) {
      v_permute_1[i0*32 + i1*1] = v_args_1[i1*64 + i0];
    }
  }
  for (int i = 0; i < 2048; i++) v_view_1[i] = v_permute_1[i];
  for (int i0 = 0; i0 < 32; i0++) {
    for (int i1 = 0; i1 < 32; i1++) {
      float acc = 0;
      for (int k = 0; k < 64; k++) {
        acc = acc + v_view[i0*64 + k] * v_view_1[k*32 + i1];
      }
    v_mm[i0*32 + i1] = (_Float16)acc;
    }
  }
  for (int i = 0; i < 1024; i++) v_view_2[i] = v_mm[i];
  for (int i = 0; i < 1024; i++) { out0[i] = v_view_2[i]; }
}
