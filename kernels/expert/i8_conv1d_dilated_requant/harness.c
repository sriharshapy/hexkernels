#include "harness_common.h"
#include "kernel_api.h"

/* n=512, ntaps=7. Dilation swept {1,2,3} X quant params swept {set0,set1}.
 * Max xlen = n+(ntaps-1)*3 = 512+18 = 530.
 * Total sweeps = 3 dilations x 2 quant sets = 6 runs.
 * Each run: 512 outputs. Total checked: 3072. Well within 60s budget. */
#define N        512
#define NTAPS    7
#define XLEN_MAX (N + (NTAPS - 1) * 3)   /* 530 */

static int8_t  x_buf[XLEN_MAX] HVX_ALIGN;
static int8_t  taps_buf[NTAPS]  HVX_ALIGN;
static int8_t  out_buf[N]       HVX_ALIGN;
static int8_t  ref_buf[N]       HVX_ALIGN;

static const int32_t MULTS[]  = {  4,  1 };
static const int     SHIFTS[] = {  5,  0 };
static const int8_t  ZPS[]    = {  0,  7 };
#define NQUANT ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

static int8_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static int run_one(int d, int qidx, int *total_errors, int *fb_out, long *gotv, long *expv) {
    int xlen = N + (NTAPS - 1) * d;
    int32_t mult  = MULTS[qidx];
    int     shift = SHIFTS[qidx];
    int8_t  zp    = ZPS[qidx];

    /* Fresh seeded inputs per (dilation, qidx) combo */
    uint32_t s = 0xC7A5E3u + (uint32_t)d * 0x1F3Bu + (uint32_t)qidx * 0x7D91u;
    for (int i = 0; i < xlen;  i++) x_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < NTAPS; j++) taps_buf[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Max-magnitude products */
    x_buf[0]       = -128; taps_buf[0] = -128;
    x_buf[d]       =  127; taps_buf[1] = -128;
    x_buf[xlen-1]  = -128;

    /* Build reference */
    for (int i = 0; i < N; i++) {
        int32_t acc = 0;
        for (int k = 0; k < NTAPS; k++)
            acc += (int32_t)x_buf[i + k * d] * (int32_t)taps_buf[k];
        ref_buf[i] = requant(acc, mult, shift, zp);
    }

    /* Poison */
    for (int i = 0; i < N; i++) out_buf[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, taps_buf, out_buf, N, NTAPS, d, mult, shift, zp); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (out_buf[i] != ref_buf[i]) {
            errors++;
            if (*fb_out < 0) {
                *fb_out = i;
                *gotv = (long)(int8_t)out_buf[i];
                *expv = (long)(int8_t)ref_buf[i];
            }
        }
    }
    *total_errors += errors;
    return errors;
}

int main(void) {
    int total_errors = 0, fb = -1; long gotv = 0, expv = 0;
    int dilations[] = {1, 2, 3};
    for (int di = 0; di < 3; di++) {
        for (int qi = 0; qi < NQUANT; qi++) {
            if (run_one(dilations[di], qi, &total_errors, &fb, &gotv, &expv) != 0 && fb >= 0) {
                /* report first failure; continue counting */
            }
        }
    }
    hvx_report(total_errors, N * 3 * NQUANT, fb, gotv, expv);
    return total_errors ? 1 : 0;
}
