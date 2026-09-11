#include "harness_common.h"
#include "kernel_api.h"

/* Dims: 12*12*8*8*9 = ~83k multiply-adds per sweep -- fine. */
#define H      8
#define W      8
#define C_IN    8
#define C_OUT   8

static int8_t  in_buf[H*W*C_IN]           HVX_ALIGN;
static int8_t  wt_buf[C_OUT*9*C_IN]       HVX_ALIGN;
static int32_t bias_buf[C_OUT]             HVX_ALIGN;
static int32_t bn_scale_buf[C_OUT]         HVX_ALIGN;
static int     bn_shift_buf[C_OUT];
static int8_t  out_buf[H*W*C_OUT]          HVX_ALIGN;
static int8_t  ref_buf[H*W*C_OUT]          HVX_ALIGN;

static int8_t ref_element(int y, int x, int co, int8_t zp) {
    int32_t acc = 0;
    for (int ky = 0; ky < 3; ky++) {
        int sy = y + ky - 1;
        for (int kx = 0; kx < 3; kx++) {
            int sx = x + kx - 1;
            for (int ci = 0; ci < C_IN; ci++) {
                int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                            ? in_buf[sy*W*C_IN + sx*C_IN + ci] : (int8_t)0;
                int8_t wv = wt_buf[co*9*C_IN + ky*3*C_IN + kx*C_IN + ci];
                acc += (int32_t)iv * (int32_t)wv;
            }
        }
    }
    int64_t biased = (int64_t)acc + (int64_t)bias_buf[co];
    int64_t v      = biased * (int64_t)bn_scale_buf[co];
    int shift      = bn_shift_buf[co];
    int64_t half   = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r      = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Two (scale, shift, zp) sweep sets to bust any hardcoder. */
static const int32_t SCALES_A[C_OUT] = { 3, 5, 2, 7, 1, 4, 6, 3 };
static const int     SHIFTS_A[C_OUT] = { 4, 5, 3, 6, 2, 4, 5, 3 };
static const int8_t  ZP_A = 0;

static const int32_t SCALES_B[C_OUT] = { 2, 3, 7, 1, 5, 2, 4, 6 };
static const int     SHIFTS_B[C_OUT] = { 3, 4, 5, 2, 6, 3, 4, 5 };
static const int8_t  ZP_B = -5;

int main(void) {
    uint32_t s = 0xABCD01u;
    for (int i = 0; i < H*W*C_IN;    i++) in_buf[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_OUT*9*C_IN; i++) wt_buf[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_OUT;        i++) bias_buf[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge: corner pixels, large negative bias ch0, large positive ch1 */
    in_buf[0] = 127; in_buf[1] = -128;
    bias_buf[0] = -100000;
    bias_buf[1] =  100000;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    /* Sweep A */
    for (int co = 0; co < C_OUT; co++) {
        bn_scale_buf[co] = SCALES_A[co];
        bn_shift_buf[co] = SHIFTS_A[co];
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int co = 0; co < C_OUT; co++)
                ref_buf[y*W*C_OUT + x*C_OUT + co] = ref_element(y, x, co, ZP_A);
    for (int i = 0; i < H*W*C_OUT; i++) out_buf[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, {
        candidate_kernel(in_buf, wt_buf, bias_buf, bn_scale_buf, bn_shift_buf, ZP_A, out_buf,
        H, W, C_IN, C_OUT);
    });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    for (int i = 0; i < H*W*C_OUT; i++) {
        if (out_buf[i] != ref_buf[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out_buf[i]; expv = (long)ref_buf[i]; }
        }
    }

    /* Sweep B */
    for (int co = 0; co < C_OUT; co++) {
        bn_scale_buf[co] = SCALES_B[co];
        bn_shift_buf[co] = SHIFTS_B[co];
    }
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int co = 0; co < C_OUT; co++)
                ref_buf[y*W*C_OUT + x*C_OUT + co] = ref_element(y, x, co, ZP_B);
    for (int i = 0; i < H*W*C_OUT; i++) out_buf[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc_b = 0;
    HVX_TIME_KERNEL(_hvx_kc_b, {
        candidate_kernel(in_buf, wt_buf, bias_buf, bn_scale_buf, bn_shift_buf, ZP_B, out_buf,
        H, W, C_IN, C_OUT);
    });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc_b);
    for (int i = 0; i < H*W*C_OUT; i++) {
        if (out_buf[i] != ref_buf[i]) {
            errors++;
            if (fb < 0) { fb = H*W*C_OUT + i; gotv = (long)out_buf[i]; expv = (long)ref_buf[i]; }
        }
    }

    hvx_report(errors, 2*H*W*C_OUT, fb, gotv, expv);
    return errors ? 1 : 0;
}
