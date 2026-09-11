/* Plain scalar int8 3x3 conv + fused bias/ReLU/saturating-requant-to-uint8 --
 * DENOMINATOR baseline. Direct nested-loop convolution (no im2col staging, no
 * HVX/HMX). Transcribed literally from harness.c's independent scalar golden
 * (the p=0 refcheck loop generalized to all output positions). */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline uint8_t saturate_u8(int v) {
    if (v > 255) v = 255;
    if (v <   0) v = 0;
    return (uint8_t)v;
}

void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      uint8_t *out, int n) {
    for (int p = 0; p < P_OUT; p++) {
        int oh = p / OW, ow = p % OW;
        for (int co = 0; co < C_OUT; co++) {
            int acc = 0;
            for (int ci = 0; ci < C_IN; ci++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        acc += (int)in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)]
                             * (int)W[co*KK + ci*(KH*KW) + kh*KW + kw];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            out[co*P_OUT + p] = saturate_u8(relu);
        }
    }
    (void)n;
}
