/* Near-miss: wrong axis order — treats output as [C][H][W] instead of [H][W][C].
   Places out[c*H*W + h*W + w] = in[...] which is just a copy, not a layout change. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                out[c*H*W + h*W + w] = in[c*H*W + h*W + w]; /* BUG: just copies, no reorder */
}
