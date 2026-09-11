/* Near-miss: wrong block ordering — bw and bh swapped in c_in decomposition.
   For b=2: c_in becomes c*4 + bh*2 + bw (correct) vs c*4 + bw*2 + bh (wrong),
   which swaps the contribution of row and column blocks. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C_in, int H_in, int W_in, int b) {
    int C_out = C_in / (b * b);
    int H_out = H_in * b;
    int W_out = W_in * b;
    for (int c = 0; c < C_out; c++)
        for (int bh = 0; bh < b; bh++)
            for (int bw = 0; bw < b; bw++) {
                int c_in = c*b*b + bw*b + bh;  /* BUG: bw and bh swapped */
                for (int oh = 0; oh < H_in; oh++)
                    for (int ow = 0; ow < W_in; ow++)
                        out[c*H_out*W_out + (oh*b + bh)*W_out + (ow*b + bw)] =
                            in[c_in * H_in*W_in + oh*W_in + ow];
            }
}
