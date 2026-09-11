/* Plain scalar int8 3x3 conv (VALID) -- the DENOMINATOR baseline. Direct
 * nested-loop convolution, no im2col staging, no HVX/HMX. Transcribed
 * literally from harness.c's reference computation. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n) {
    for (int co = 0; co < C_OUT; co++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int acc = 0;
                for (int ci = 0; ci < C_IN; ci++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            acc += (int)in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)]
                                 * (int)W[((co*C_IN + ci)*KH + kh)*KW + kw];
                out[co*P_OUT + oh*OW + ow] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }
    (void)n;
}
