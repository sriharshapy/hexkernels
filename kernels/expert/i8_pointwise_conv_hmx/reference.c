/* Baseline: pure scalar int8 1x1 (pointwise) convolution — the DENOMINATOR
 * baseline. out[co][p] = requant_0x40( sum_ci in[ci*P+p] * W[co*C_in+ci] ).
 * in is uint8 0..7, W is int8; K=C_in=96 keeps the int accumulator well within
 * range (no overflow). Same 0x40-config requant as the reference. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n) {
    (void)n;
    for (int co = 0; co < CONV_COUT; co++)
        for (int p = 0; p < CONV_P; p++) {
            int acc = 0;
            for (int ci = 0; ci < CONV_CIN; ci++)
                acc += (int)in[ci*CONV_P + p] * (int)W[co*CONV_CIN + ci];
            out[co*CONV_P + p] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
}
