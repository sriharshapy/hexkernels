/* Near-miss: hardcodes dw and pw quant params (uses first set's values).
   Fails when second sweep set has different mult/shift/zp. */
#include <stdint.h>
void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt,
                      const int8_t *pw_wt,
                      int8_t *mid, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    (void)dw_mult; (void)dw_shift; (void)dw_zp;
    (void)pw_mult; (void)pw_shift; (void)pw_zp;
    /* Stage 1: depthwise -- hardcoded dw mult=3, shift=4, zp=0 */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C_in; c++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                    ? in[sy*W*C_in + sx*C_in + c] : (int8_t)0;
                        int8_t wv = dw_wt[c*9 + ky*3 + kx];
                        acc += (int32_t)iv * (int32_t)wv;
                    }
                }
                long long v = (long long)acc * 3LL;
                long long h = 8LL;
                long long r = (v >= 0) ? ((v + h) >> 4) : -((-v + h) >> 4);
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                mid[y*W*C_in + x*C_in + c] = (int8_t)r;
            }
        }
    }
    /* Stage 2: pointwise -- hardcoded pw mult=2, shift=3, zp=0 */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int co = 0; co < C_out; co++) {
                int32_t acc = 0;
                for (int ci = 0; ci < C_in; ci++)
                    acc += (int32_t)mid[y*W*C_in + x*C_in + ci]
                         * (int32_t)pw_wt[co*C_in + ci];
                long long v = (long long)acc * 2LL;
                long long h = 4LL;
                long long r = (v >= 0) ? ((v + h) >> 3) : -((-v + h) >> 3);
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out[y*W*C_out + x*C_out + co] = (int8_t)r;
            }
        }
    }
}
