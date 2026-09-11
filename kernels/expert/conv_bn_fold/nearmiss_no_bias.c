/* Near-miss: skips the bias addition -- applies bn_scale/bn_shift directly on raw acc.
   The bias (folded bn offset) is ignored, producing wrong outputs whenever bias != 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      const int32_t *bias,
                      const int32_t *bn_scale, const int *bn_shift,
                      int8_t zp,
                      int8_t *out,
                      int H, int W, int C_in, int C_out) {
    (void)bias; /* BUG: ignore bias */
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
                /* no bias */
                int64_t v    = (int64_t)acc * (int64_t)bn_scale[co];
                int     shift = bn_shift[co];
                int64_t half  = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
                int64_t r     = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
                r += zp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out[y*W*C_out + x*C_out + co] = (int8_t)r;
            }
        }
    }
}
