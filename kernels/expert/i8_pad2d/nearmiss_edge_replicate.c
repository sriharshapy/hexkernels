/* Near-miss: edge-replicate padding instead of zero-padding.
   Border pixels copy the nearest edge pixel instead of writing 0.
   Compiles and runs but is incorrect (non-zero border). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out,
                      int H, int W, int P) {
    int OH = H + 2*P, OW = W + 2*P;
    for (int r = 0; r < OH; r++) {
        for (int c = 0; c < OW; c++) {
            /* clamp row and col to [0, H-1] x [0, W-1] — BUG: should zero-fill border */
            int sr = r - P;
            int sc = c - P;
            if (sr < 0) sr = 0;
            if (sr >= H) sr = H - 1;
            if (sc < 0) sc = 0;
            if (sc >= W) sc = W - 1;
            out[r * OW + c] = in[sr * W + sc];
        }
    }
}
