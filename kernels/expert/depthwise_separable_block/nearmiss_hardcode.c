/* Near-miss: correct DW+PW pipeline but hardcodes dw: mult=3,shift=6,zp=0
   and pw: mult=5,shift=7,zp=0. Passes first sweep set but FAILS the other 3. */
#include <stdint.h>
static int8_t rq_hc(int32_t val, long long mult, int shift) {
    long long v = (long long)val * mult;
    long long h = shift > 0 ? (1LL << (shift - 1)) : 0;
    long long r = (v >= 0) ? ((v + h) >> shift) : -((-v + h) >> shift);
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (signed char)r;
}
void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt, const int32_t *dw_bias,
                      const int8_t *pw_wt, const int32_t *pw_bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    (void)dw_mult; (void)dw_shift; (void)dw_zp;
    (void)pw_mult; (void)pw_shift; (void)pw_zp;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            signed char dw_q[64];
            for (int c = 0; c < C_in; c++) {
                int acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        signed char iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                         ? in[sy*W*C_in + sx*C_in + c] : 0;
                        acc += (int)iv * (int)dw_wt[c*9 + ky*3 + kx];
                    }
                }
                int biased = acc + dw_bias[c];
                if (biased < 0) biased = 0;
                dw_q[c] = rq_hc(biased, 3LL, 6);  /* hardcoded dw: mult=3, shift=6 */
            }
            for (int co = 0; co < C_out; co++) {
                int pw = 0;
                for (int c = 0; c < C_in; c++)
                    pw += (int)dw_q[c] * (int)pw_wt[co*C_in + c];
                int biased = pw + pw_bias[co];
                if (biased < 0) biased = 0;
                out[y*W*C_out + x*C_out + co] = rq_hc(biased, 5LL, 7);  /* hardcoded pw: mult=5, shift=7 */
            }
        }
    }
}
