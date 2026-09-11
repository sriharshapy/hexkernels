#include <stdint.h>
/* NCHW->NHWC: out[h*W*C + w*C + c] = in[c*H*W + h*W + w] */
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                out[h*W*C + w*C + c] = in[c*H*W + h*W + w];
}
