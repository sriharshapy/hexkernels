#include <stdint.h>
#include <math.h>

static float v__convolution[1024];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, float *out0) {
  for (int n = 0; n < 1; n++) {
    for (int oc = 0; oc < 16; oc++) {
      for (int oh = 0; oh < 8; oh++) {
        for (int ow = 0; ow < 8; ow++) {
          float acc = 0;
          int g = oc / 16;
          for (int ic = 0; ic < 16; ic++) {
            int in_c = g * 16 + ic;
            for (int kh = 0; kh < 2; kh++) {
              for (int kw = 0; kw < 2; kw++) {
                int ih = oh*2 - 0 + kh*1;
                int iw = ow*2 - 0 + kw*1;
                if (ih >= 0 && ih < 16 && iw >= 0 && iw < 16) {
                  acc = acc + v_args_0[n*4096 + in_c*256 + ih*16 + iw*1] * v_args_1[oc*64 + ic*4 + kh*2 + kw*1];
                }
              }
            }
          }
          v__convolution[n*1024 + oc*64 + oh*8 + ow*1] = acc;
        }
      }
    }
  }
  for (int i = 0; i < 1024; i++) { out0[i] = v__convolution[i]; }
}
