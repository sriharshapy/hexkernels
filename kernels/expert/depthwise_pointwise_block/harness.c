#include "harness_common.h"
#include "kernel_api.h"

/* Small dims: DW=O(H*W*C_in*9), PW=O(H*W*C_in*C_out).
   12*12*8*9 + 12*12*8*8 = 10368+9216 = ~20k ops per sweep. Safe. */
#define H      12
#define W      12
#define C_IN    8
#define C_OUT   8

static int8_t in_buf[H*W*C_IN]        HVX_ALIGN;
static int8_t dw_wt_buf[C_IN*9]       HVX_ALIGN;
static int8_t pw_wt_buf[C_OUT*C_IN]   HVX_ALIGN;
static int8_t mid_buf[H*W*C_IN]       HVX_ALIGN;
static int8_t out_buf[H*W*C_OUT]      HVX_ALIGN;
static int8_t ref_mid[H*W*C_IN]       HVX_ALIGN;
static int8_t ref_out[H*W*C_OUT]      HVX_ALIGN;

static int8_t requant8(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static void build_ref(int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp) {
    /* Stage 1: depthwise */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C_IN; c++) {
                int32_t acc = 0;
                for (int ky = 0; ky < 3; ky++) {
                    int sy = y + ky - 1;
                    for (int kx = 0; kx < 3; kx++) {
                        int sx = x + kx - 1;
                        int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                                    ? in_buf[sy*W*C_IN + sx*C_IN + c] : (int8_t)0;
                        int8_t wv = dw_wt_buf[c*9 + ky*3 + kx];
                        acc += (int32_t)iv * (int32_t)wv;
                    }
                }
                ref_mid[y*W*C_IN + x*C_IN + c] = requant8(acc, dw_mult, dw_shift, dw_zp);
            }
        }
    }
    /* Stage 2: pointwise */
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int co = 0; co < C_OUT; co++) {
                int32_t acc = 0;
                for (int ci = 0; ci < C_IN; ci++)
                    acc += (int32_t)ref_mid[y*W*C_IN + x*C_IN + ci]
                         * (int32_t)pw_wt_buf[co*C_IN + ci];
                ref_out[y*W*C_OUT + x*C_OUT + co] = requant8(acc, pw_mult, pw_shift, pw_zp);
            }
        }
    }
}

/* Two sweep sets */
static const int32_t DW_MULTS[]  = {  3,  7 };
static const int     DW_SHIFTS[] = {  4,  5 };
static const int8_t  DW_ZPS[]    = {  0, -2 };
static const int32_t PW_MULTS[]  = {  2,  4 };
static const int     PW_SHIFTS[] = {  3,  6 };
static const int8_t  PW_ZPS[]    = {  0,  3 };
#define NSETS 2

int main(void) {
    uint32_t s = 0xC0DE77u;
    for (int i = 0; i < H*W*C_IN;    i++) in_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_IN*9;      i++) dw_wt_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_OUT*C_IN;  i++) pw_wt_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge: corners exercise zero-padding in depthwise */
    in_buf[0] = 127; in_buf[C_IN - 1] = -128;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        build_ref(DW_MULTS[p], DW_SHIFTS[p], DW_ZPS[p],
                  PW_MULTS[p], PW_SHIFTS[p], PW_ZPS[p]);

        for (int i = 0; i < H*W*C_IN;  i++) mid_buf[i] = (int8_t)0xA5;
        for (int i = 0; i < H*W*C_OUT; i++) out_buf[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, {
            candidate_kernel(in_buf, dw_wt_buf, pw_wt_buf, mid_buf, out_buf,
            H, W, C_IN, C_OUT,
            DW_MULTS[p], DW_SHIFTS[p], DW_ZPS[p],
            PW_MULTS[p], PW_SHIFTS[p], PW_ZPS[p]);
        });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < H*W*C_OUT; i++) {
            if (out_buf[i] != ref_out[i]) {
                errors++;
                if (fb < 0) { fb = p*H*W*C_OUT + i; gotv = (long)out_buf[i]; expv = (long)ref_out[i]; }
            }
        }
    }
    hvx_report(errors, H*W*C_OUT*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
