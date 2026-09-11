/* DENOMINATOR baseline — pure scalar depthwise (per-channel, NHWC) + pointwise
 * (1x1) + fused bias/ReLU/saturating-requant. No HVX/HMX/DMA/VTCM: plain
 * nested loops, matching the harness's own scalar reference computation. */
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

void candidate_kernel(const uint8_t *in, const int8_t *Wdw, const int8_t *Wpw,
                      const int32_t *bias, uint8_t *out, int n) {
    static uint8_t dw[P_OUT*C_IN];   /* depthwise result, ReLU'd, 0..27 */

    /* --- depthwise (VALID 3x3, per-channel) --- */
    for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
            for (int c = 0; c < C_IN; c++) {
                int acc = 0;
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        acc += (int)in[((oh*STRIDE + kh)*IW + (ow*STRIDE + kw))*C_IN + c]
                             * (int)Wdw[c*(KH*KW) + kh*KW + kw];
                dw[(oh*OW + ow)*C_IN + c] = saturate_u8(acc > 0 ? acc : 0);
            }

    /* --- pointwise (1x1) + fused bias/ReLU/saturating-requant --- */
    for (int p = 0; p < P_OUT; p++) {
        for (int co = 0; co < C_OUT; co++) {
            int acc = 0;
            for (int c = 0; c < C_IN; c++) acc += (int)dw[p*C_IN + c] * (int)Wpw[co*C_IN + c];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[co];
            int relu = biased > 0 ? biased : 0;
            out[p*C_OUT + co] = saturate_u8(relu);
        }
    }
    (void)n;
}
