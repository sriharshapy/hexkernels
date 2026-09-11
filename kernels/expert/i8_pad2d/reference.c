#include <stdint.h>
/*
 * Zero-pad 2D: out[r*(W+2P)+c] = in[(r-P)*W+(c-P)] if in-bounds, else 0.
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int H, int W, int P) {
    int OH = H + 2*P, OW = W + 2*P;
    for (int r = 0; r < OH; r++) {
        for (int c = 0; c < OW; c++) {
            int sr = r - P, sc = c - P;
            out[r * OW + c] = (sr >= 0 && sr < H && sc >= 0 && sc < W)
                              ? in[sr * W + sc]
                              : (int8_t)0;
        }
    }
}
