#include "harness_common.h"
#include "kernel_api.h"
#define H      8
#define W      8
#define C_IN   8
#define C_OUT  8

static int8_t  in_buf[H*W*C_IN]      HVX_ALIGN;
static int8_t  dw_wt_buf[C_IN*9]     HVX_ALIGN;  /* [C_in][3][3] */
static int32_t dw_bias_buf[C_IN]      HVX_ALIGN;
static int8_t  pw_wt_buf[C_OUT*C_IN]  HVX_ALIGN;  /* [C_out][C_in] */
static int32_t pw_bias_buf[C_OUT]     HVX_ALIGN;
static int8_t  out_buf[H*W*C_OUT]    HVX_ALIGN;
static int8_t  ref_buf[H*W*C_OUT]    HVX_ALIGN;

/* Helper: requantize a non-negative int32 (after relu) to int8 */
static int8_t requant(int32_t val, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)val * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static int8_t ref_element(int y, int x, int co,
                           int32_t dw_mult, int dw_shift, int8_t dw_zp,
                           int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    /* DW stage */
    int8_t dw_q[C_IN];
    for (int c = 0; c < C_IN; c++) {
        int32_t acc = 0;
        for (int ky = 0; ky < 3; ky++) {
            int sy = y + ky - 1;
            for (int kx = 0; kx < 3; kx++) {
                int sx = x + kx - 1;
                int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                            ? in_buf[sy*W*C_IN + sx*C_IN + c] : (int8_t)0;
                acc += (int32_t)iv * (int32_t)dw_wt_buf[c*9 + ky*3 + kx];
            }
        }
        int32_t biased = acc + dw_bias_buf[c];
        if (biased < 0) biased = 0;  /* relu */
        dw_q[c] = requant(biased, dw_mult, dw_shift, dw_zp);
    }
    /* PW stage */
    int32_t pw = 0;
    for (int c = 0; c < C_IN; c++)
        pw += (int32_t)dw_q[c] * (int32_t)pw_wt_buf[co*C_IN + c];
    int32_t pw_biased = pw + pw_bias_buf[co];
    if (pw_biased < 0) pw_biased = 0;  /* relu */
    return requant(pw_biased, pw_mult, pw_shift, pw_zp);
}

/* Sweep DW and PW quant params together. */
static const int32_t DW_MULTS[] = { 3, 1 };
static const int     DW_SHIFTS[]= { 6, 0 };
static const int     DW_ZPS[]   = { 0, 0 };
static const int32_t PW_MULTS[] = { 5, 2 };
static const int     PW_SHIFTS[]= { 7, 0 };
static const int     PW_ZPS[]   = { 0, -2 };
#define NSETS ((int)(sizeof(DW_MULTS)/sizeof(DW_MULTS[0])))

int main(void) {
    uint32_t s = 0xC0FFEE1u;
    for (int i = 0; i < H*W*C_IN;   i++) in_buf[i]      = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_IN*9;     i++) dw_wt_buf[i]   = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_IN;       i++) dw_bias_buf[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < C_OUT*C_IN; i++) pw_wt_buf[i]   = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_OUT;      i++) pw_bias_buf[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    /* Edge cases: large neg DW bias forces relu; corner pixels exercise padding */
    in_buf[0] = 127; in_buf[C_IN-1] = -128;
    dw_bias_buf[0] = -500000;   /* ch0 DW relu fires strongly */
    pw_bias_buf[0] = -500000;   /* co0 PW relu fires strongly */
    pw_bias_buf[1] =  500000;   /* co1 large positive */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        int32_t dm = DW_MULTS[p]; int ds = DW_SHIFTS[p]; int8_t dz = (int8_t)DW_ZPS[p];
        int32_t pm = PW_MULTS[p]; int ps = PW_SHIFTS[p]; int8_t pz = (int8_t)PW_ZPS[p];

        /* Build reference */
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int co = 0; co < C_OUT; co++)
                    ref_buf[y*W*C_OUT + x*C_OUT + co] =
                        ref_element(y, x, co, dm, ds, dz, pm, ps, pz);

        /* Poison output */
        for (int i = 0; i < H*W*C_OUT; i++) out_buf[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, {
            candidate_kernel(in_buf, dw_wt_buf, dw_bias_buf, pw_wt_buf, pw_bias_buf, out_buf,
            H, W, C_IN, C_OUT, dm, ds, dz, pm, ps, pz);
        });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < H*W*C_OUT; i++) {
            if (out_buf[i] != ref_buf[i]) {
                errors++;
                if (fb < 0) {
                    fb = p*H*W*C_OUT + i;
                    gotv = (long)out_buf[i];
                    expv = (long)ref_buf[i];
                }
            }
        }
    }
    hvx_report(errors, H*W*C_OUT*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
