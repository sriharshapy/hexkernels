/* Near-miss: correct depthwise conv + bias + requant, but SKIPS ReLU.
   Fails wherever acc+bias < 0 (which happens for channel 0 with bias_buf[0]=-200000). */
#include <stdint.h>
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp) {
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
                int64_t biased = (int64_t)acc + (int64_t)bias[c];
                /* BUG: no relu */
                int64_t v    = biased * (int64_t)mult;
                int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
                int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
                r += zp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                out[y*W*C + x*C + c] = (int8_t)r;
            }
        }
    }
}
