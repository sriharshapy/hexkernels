#include <stdint.h>
static int8_t requant8(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt,
                      const int8_t *pw_wt,
                      int8_t *mid, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    /* Stage 1: depthwise conv + requant -> mid */
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
                mid[y*W*C_in + x*C_in + c] = requant8(acc, dw_mult, dw_shift, dw_zp);
            }
        }
    }
    /* Stage 2: pointwise conv + requant -> out */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int co = 0; co < C_out; co++) {
                int32_t acc = 0;
                for (int ci = 0; ci < C_in; ci++)
                    acc += (int32_t)mid[y*W*C_in + x*C_in + ci]
                         * (int32_t)pw_wt[co*C_in + ci];
                out[y*W*C_out + x*C_out + co] = requant8(acc, pw_mult, pw_shift, pw_zp);
            }
        }
    }
}
