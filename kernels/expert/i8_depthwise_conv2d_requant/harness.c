#include "harness_common.h"
#include "kernel_api.h"

/* Small dims: depthwise is O(H*W*C*9) -- 14*14*8*9 = ~14k ops per sweep.
   2 sweep sets: fast enough in functional sim. */
#define H    14
#define W    14
#define C     8

static int8_t  in_buf[H*W*C]  HVX_ALIGN;
static int8_t  wt_buf[C*9]    HVX_ALIGN;
static int8_t  out_buf[H*W*C] HVX_ALIGN;
static int8_t  ref_buf[H*W*C] HVX_ALIGN;

static int8_t ref_element(int y, int x, int c,
                           int32_t mult, int shift, int8_t zp) {
    int32_t acc = 0;
    for (int ky = 0; ky < 3; ky++) {
        int sy = y + ky - 1;
        for (int kx = 0; kx < 3; kx++) {
            int sx = x + kx - 1;
            int8_t iv = (sy >= 0 && sy < H && sx >= 0 && sx < W)
                        ? in_buf[sy*W*C + sx*C + c] : (int8_t)0;
            int8_t wv = wt_buf[c*9 + ky*3 + kx];
            acc += (int32_t)iv * (int32_t)wv;
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

/* Two sweep sets: bust any hardcoder of mult/shift/zp. */
static const int32_t MULTS[]  = {  5,  2 };
static const int     SHIFTS[] = {  4,  6 };
static const int     ZPS[]    = {  0, -4 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xBEEF13u;
    for (int i = 0; i < H*W*C; i++) in_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < C*9;   i++) wt_buf[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge: corner pixels exercise zero-padding */
    in_buf[0] = 127; in_buf[C-1] = -128;
    /* Force large positive and negative accs on channel 0 */
    wt_buf[0] = 127; wt_buf[1] = -128;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        int32_t mult  = MULTS[p];
        int     shift = SHIFTS[p];
        int8_t  zp    = (int8_t)ZPS[p];

        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int c = 0; c < C; c++)
                    ref_buf[y*W*C + x*C + c] = ref_element(y, x, c, mult, shift, zp);

        for (int i = 0; i < H*W*C; i++) out_buf[i] = (int8_t)0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in_buf, wt_buf, out_buf, H, W, C, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < H*W*C; i++) {
            if (out_buf[i] != ref_buf[i]) {
                errors++;
                if (fb < 0) { fb = p*H*W*C + i; gotv = (long)out_buf[i]; expv = (long)ref_buf[i]; }
            }
        }
    }
    hvx_report(errors, H*W*C*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
