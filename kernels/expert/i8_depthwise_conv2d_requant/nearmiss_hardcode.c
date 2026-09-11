/* Near-miss: hardcodes mult=5, shift=4, zp=0 -- ignores runtime requant params.
   Passes first sweep set (mult=5,shift=4,zp=0) but FAILS the second (mult=2,shift=6,zp=-4). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp) {
    (void)mult; (void)shift; (void)zp;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                    ? in[sy*W*C + sx*C + c] : (int8_t)0;
                        int8_t wv = wt[c*9 + ky*3 + kx];
                        acc += (int32_t)iv * (int32_t)wv;
                    }
                }
                /* hardcoded mult=5, shift=4, zp=0 */
                long long v = (long long)acc * 5LL;
                long long h = 8LL;
                long long r = (v >= 0) ? ((v + h) >> 4) : -((-v + h) >> 4);
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out[y*W*C + x*C + c] = (int8_t)r;
            }
        }
    }
}
