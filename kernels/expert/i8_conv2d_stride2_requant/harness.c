#include "harness_common.h"
#include "kernel_api.h"

/* Stride-2: output is H/2 x W/2. Use H=W=16 (even), C_in=C_out=8.
   Output: 8*8*8 = 512 elements. Conv ops per sweep: 8*8*8*9*8 = ~37k. Fine. */
#define H      16
#define W      16
#define OUT_H  (H/2)
#define OUT_W  (W/2)
#define C_IN    8
#define C_OUT   8

static int8_t  in_buf[H*W*C_IN]           HVX_ALIGN;
static int8_t  wt_buf[C_OUT*9*C_IN]       HVX_ALIGN;
static int8_t  out_buf[OUT_H*OUT_W*C_OUT]  HVX_ALIGN;
static int8_t  ref_buf[OUT_H*OUT_W*C_OUT]  HVX_ALIGN;

/* Stride-2 SAME padding: pad_top=1, pad_left=1 for f=3, stride=2, even input. */
static int8_t ref_element(int oy, int ox, int co,
                           int32_t mult, int shift, int8_t zp) {
    int iy_base = oy * 2;
    int ix_base = ox * 2;
    int32_t acc = 0;
    for (int ky = 0; ky < 3; ky++) {
        int sy = iy_base + ky - 1;
        for (int kx = 0; kx < 3; kx++) {
            int sx = ix_base + kx - 1;
            for (int ci = 0; ci < C_IN; ci++) {
                int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                            ? in_buf[sy*W*C_IN + sx*C_IN + ci] : (int8_t)0;
                int8_t wv = wt_buf[co*9*C_IN + ky*3*C_IN + kx*C_IN + ci];
                acc += (int32_t)iv * (int32_t)wv;
            }
        }
    }
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static const int32_t MULTS[]  = {  3,  5 };
static const int     SHIFTS[] = {  4,  6 };
static const int     ZPS[]    = {  0, -6 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xDEAD55u;
    for (int i = 0; i < H*W*C_IN;     i++) in_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C_OUT*9*C_IN; i++) wt_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge: corner pixels exercise padding. First stride output uses pad. */
    in_buf[0] = 127; in_buf[1] = -128;
    /* Last row corner */
    in_buf[(H-1)*W*C_IN] = 127;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        int32_t mult = MULTS[p]; int shift = SHIFTS[p]; int8_t zp = (int8_t)ZPS[p];

        for (int oy = 0; oy < OUT_H; oy++)
            for (int ox = 0; ox < OUT_W; ox++)
                for (int co = 0; co < C_OUT; co++)
                    ref_buf[oy*OUT_W*C_OUT + ox*C_OUT + co] = ref_element(oy, ox, co, mult, shift, zp);

        for (int i = 0; i < OUT_H*OUT_W*C_OUT; i++) out_buf[i] = (int8_t)0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in_buf, wt_buf, out_buf, H, W, C_IN, C_OUT, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < OUT_H*OUT_W*C_OUT; i++) {
            if (out_buf[i] != ref_buf[i]) {
                errors++;
                if (fb < 0) { fb = p*OUT_H*OUT_W*C_OUT + i; gotv = (long)out_buf[i]; expv = (long)ref_buf[i]; }
            }
        }
    }
    hvx_report(errors, OUT_H*OUT_W*C_OUT*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
