#include <stdint.h>
/* NHWC->NCHW: out[c*H*W + h*W + w] = in[h*W*C + w*C + c] */
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++)
                out[c*H*W + h*W + w] = in[h*W*C + w*C + c];
}
