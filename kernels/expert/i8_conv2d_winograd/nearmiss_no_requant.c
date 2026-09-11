/* Near-miss: correct direct-conv accumulation but truncates acc to int8
   directly (no requantize step). Gives wrong values whenever |acc| > 127
   or mult/shift/zp would change the value. Fails all 4 sweep sets. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, const int8_t *wt, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int co = 0; co < C_out; co++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        for (int ci = 0; ci < C_in; ci++) {
                            int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                        ? in[sy*W*C_in + sx*C_in + ci] : (int8_t)0;
                            int8_t wv = wt[co*9*C_in + ky*3*C_in + kx*C_in + ci];
                            acc += (int32_t)iv * (int32_t)wv;
                        }
                    }
                }
                /* BUG: drops requantize, clamps raw acc to int8 range */
                if (acc >  127) acc =  127;
                if (acc < -128) acc = -128;
                out[y*W*C_out + x*C_out + co] = (int8_t)acc;
            }
        }
    }
}
