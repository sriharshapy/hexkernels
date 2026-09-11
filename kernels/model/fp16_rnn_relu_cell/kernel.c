#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <hmx_hexagon_protos.h>

static _Float16 transpose_hh[384 * 320];
static _Float16 transpose_ih[384 * 320];

extern "C" void candidate_kernel(
    const _Float16 *v_args_0,
    const _Float16 *v_args_1,
    const _Float16 *v_args_2,
    const _Float16 *v_args_3,
    _Float16 *out0) {
  for (int k = 0; k < 384; ++k) {
    for (int j = 0; j < 320; ++j) {
      transpose_hh[k * 320 + j] = v_args_3[j * 384 + k];
      transpose_ih[k * 320 + j] = v_args_2[j * 384 + k];
    }
  }

  for (int i = 0; i < 320; ++i) {
    for (int j = 0; j < 320; ++j) {
      volatile float acc_hh = 0.0f;
      volatile float acc_ih = 0.0f;

      for (int k = 0; k < 384; ++k) {
        acc_hh = acc_hh +
                 (float)v_args_1[i * 384 + k] *
                 (float)transpose_hh[k * 320 + j];
        acc_ih = acc_ih +
                 (float)v_args_0[i * 384 + k] *
                 (float)transpose_ih[k * 320 + j];
      }

      const _Float16 mm_hh = (_Float16)acc_hh;
      const _Float16 mm_ih = (_Float16)acc_ih;
      const _Float16 sum = mm_hh + mm_ih;
      out0[i * 320 + j] = sum > (_Float16)0 ? sum : (_Float16)0;
    }
  }
}
