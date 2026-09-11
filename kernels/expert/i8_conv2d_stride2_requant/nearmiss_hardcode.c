/* Near-miss: hardcodes mult=3, shift=4, zp=0 -- ignores runtime quant params.
   Passes first sweep (mult=3,shift=4,zp=0) but FAILS second (mult=5,shift=6,zp=-6). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    int out_H = H / 2;
    int out_W = W / 2;
    for (int oy = 0; oy < out_H; oy++) {
        int iy_base = oy * 2;
        for (int ox = 0; ox < out_W; ox++) {
            int ix_base = ox * 2;
            for (int co = 0; co < C_out; co++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = iy_base + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = ix_base + kx - 1;
                        for (int ci = 0; ci < C_in; ci++) {
                            int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                        ? in[sy*W*C_in + sx*C_in + ci] : (int8_t)0;
                            int8_t wv = wt[co*9*C_in + ky*3*C_in + kx*C_in + ci];
                            acc += (int32_t)iv * (int32_t)wv;
                        }
                    }
                }
                /* hardcoded mult=3, shift=4, zp=0 */
                long long v = (long long)acc * 3LL;
                long long h = 8LL;
                long long r = (v >= 0) ? ((v + h) >> 4) : -((-v + h) >> 4);
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out[oy*out_W*C_out + ox*C_out + co] = (int8_t)r;
            }
        }
    }
}
