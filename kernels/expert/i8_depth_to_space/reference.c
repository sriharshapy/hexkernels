#include <stdint.h>
/*
 * Depth-to-space baseline.
 * out[c*H_out*W_out + (oh*b+bh)*W_out + (ow*b+bw)] = in[c_in*H_in*W_in + oh*W_in + ow]
 * where c_in = c*b*b + bh*b + bw
 */
void candidate_kernel(const int8_t *in, int8_t *out,
                      int C_in, int H_in, int W_in, int b) {
    int C_out = C_in / (b * b);
    int H_out = H_in * b;
    int W_out = W_in * b;
    for (int c = 0; c < C_out; c++)
        for (int bh = 0; bh < b; bh++)
            for (int bw = 0; bw < b; bw++) {
                int c_in = c*b*b + bh*b + bw;
                for (int oh = 0; oh < H_in; oh++)
                    for (int ow = 0; ow < W_in; ow++)
                        out[c*H_out*W_out + (oh*b + bh)*W_out + (ow*b + bw)] =
                            in[c_in * H_in*W_in + oh*W_in + ow];
            }
}
