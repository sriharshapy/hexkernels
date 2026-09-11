#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <hmx_hexagon_protos.h>

extern "C" void candidate_kernel(const _Float16 *v_args_0,
                                const _Float16 *v_args_1,
                                _Float16 *out0) {
  _Float16 btile[128][8];

  for (int j0 = 0; j0 < 64; j0 += 8) {
    for (int k = 0; k < 128; ++k) {
      for (int jj = 0; jj < 8; ++jj) {
        btile[k][jj] = v_args_1[(j0 + jj) * 128 + k];
      }
    }

    for (int i = 0; i < 64; ++i) {
      volatile float acc[8];
      for (int jj = 0; jj < 8; ++jj) {
        acc[jj] = 0.0f;
      }

      for (int k = 0; k < 128; ++k) {
        const float a = (float)v_args_0[i * 128 + k];
        for (int jj = 0; jj < 8; ++jj) {
          volatile float product =
              a * (float)btile[k][jj];
          acc[jj] = acc[jj] + product;
        }
      }

      for (int jj = 0; jj < 8; ++jj) {
        out0[i * 64 + j0 + jj] = (_Float16)acc[jj];
      }
    }
  }
}
