#include <stdint.h>
#include <math.h>

static _Float16 v__convolution[64];

extern "C" void candidate_kernel(const _Float16 *v_args_0, const _Float16 *v_args_1, _Float16 *out0) {
  for (int n = 0; n < 1; n++) {
    for (int oc = 0; oc < 4; oc++) {
      for (int oh = 0; oh < 4; oh++) {
        for (int ow = 0; ow < 4; ow++) {
          float acc = 0;
          int g = oc / 4;
          for (int ic = 0; ic < 4; ic++) {
            int in_c = g * 4 + ic;
            for (int kh = 0; kh < 2; kh++) {
              for (int kw = 0; kw < 2; kw++) {
                int ih = oh*2 - 0 + kh*1;
                int iw = ow*2 - 0 + kw*1;
                if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8) {
                  acc = acc + v_args_0[n*256 + in_c*64 + ih*8 + iw*1] * v_args_1[oc*16 + ic*4 + kh*2 + kw*1];
                }
              }
            }
          }
          v__convolution[n*64 + oc*16 + oh*4 + ow*1] = (_Float16)acc;
        }
      }
    }
  }
  for (int i = 0; i < 64; i++) { out0[i] = v__convolution[i]; }
}
