#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <hvx_hexagon_protos.h>
#include <hmx_hexagon_protos.h>

extern "C" void candidate_kernel(const _Float16 *v_args_0,
                                 const _Float16 *v_args_1,
                                 _Float16 *out0) {
  for (int oc = 0; oc < 64; ++oc) {
    for (int oh = 0; oh < 64; ++oh) {
      for (int ow = 0; ow < 64; ++ow) {
        volatile float acc = 0.0f;

        for (int ic = 0; ic < 64; ++ic) {
          const int input_channel_base = ic * 16384;
          const int weight_base = oc * 256 + ic * 4;
          const int input_row_base = oh * 2 * 128 + ow * 2;

          for (int kh = 0; kh < 2; ++kh) {
            const int input_base =
                input_channel_base + input_row_base + kh * 128;
            const int weight_row_base = weight_base + kh * 2;

            volatile float p0 =
                (float)v_args_0[input_base] *
                (float)v_args_1[weight_row_base];
            acc = acc + p0;

            volatile float p1 =
                (float)v_args_0[input_base + 1] *
                (float)v_args_1[weight_row_base + 1];
            acc = acc + p1;
          }
        }

        out0[oc * 4096 + oh * 64 + ow] = (_Float16)acc;
      }
    }
  }
}
