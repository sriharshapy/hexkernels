/* Near-miss: wrong axis order — treats output as [H][W][C] instead of [C][H][W].
   This is just a copy, not a layout conversion. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    for (int h = 0; h < H; h++)
        for (int w = 0; w < W; w++)
            for (int c = 0; c < C; c++)
                out[h*W*C + w*C + c] = in[h*W*C + w*C + c]; /* BUG: just copies, no reorder */
}
