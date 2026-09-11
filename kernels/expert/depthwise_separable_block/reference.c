#include <stdint.h>
/* Scalar ground truth: BN-folded DSC block -- DW+bias+relu+requant -> PW+bias+relu+requant -> i8. */
static int8_t requant8(int32_t val, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)val * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt, const int32_t *dw_bias,
                      const int8_t *pw_wt, const int32_t *pw_bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            /* DW stage */
            int8_t dw_q[64]; /* max C_in */
            for (int c = 0; c < C_in; c++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                    ? in[sy*W*C_in + sx*C_in + c] : (int8_t)0;
                        acc += (int32_t)iv * (int32_t)dw_wt[c*9 + ky*3 + kx];
                    }
                }
                int32_t biased = acc + dw_bias[c];
                if (biased < 0) biased = 0;  /* relu */
                dw_q[c] = requant8(biased, dw_mult, dw_shift, dw_zp);
            }
            /* PW stage */
            for (int co = 0; co < C_out; co++) {
                int32_t pw = 0;
                for (int c = 0; c < C_in; c++)
                    pw += (int32_t)dw_q[c] * (int32_t)pw_wt[co*C_in + c];
                int32_t biased = pw + pw_bias[co];
                if (biased < 0) biased = 0;  /* relu */
                out[y*W*C_out + x*C_out + co] = requant8(biased, pw_mult, pw_shift, pw_zp);
            }
        }
    }
}
