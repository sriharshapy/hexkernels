#include <stdint.h>
/*
 * im2col baseline.
 * out[row * OH*OW + col] = in[c*H*W + (oh*S + kh)*W + (ow*S + kw)]
 * row = c*K*K + kh*K + kw,  col = oh*OW + ow
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C, int H, int W,
                      int K, int S,
                      int OH, int OW) {
    (void)H;
    for (int c = 0; c < C; c++)
        for (int kh = 0; kh < K; kh++)
            for (int kw = 0; kw < K; kw++) {
                int row = c*K*K + kh*K + kw;
                for (int oh = 0; oh < OH; oh++)
                    for (int ow = 0; ow < OW; ow++) {
                        int col = oh*OW + ow;
                        out[row * OH*OW + col] = in[c*H*W + (oh*S + kh)*W + (ow*S + kw)];
                    }
            }
}
