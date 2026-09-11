#include "harness_common.h"
#include "kernel_api.h"

/* C=8 channels, L=256 output samples/channel, ntaps=7.
   Input per channel = L + ntaps - 1 = 262 samples. */
#define C     8
#define L     256
#define NTAPS 7
#define XSTRIDE (L + NTAPS - 1)   /* 262 samples per channel in x */
#define XLEN  (C * XSTRIDE)       /* 2096 total */
#define TLEN  (C * NTAPS)         /* 56 */
#define OLEN  (C * L)             /* 2048 */

static int8_t  x_buf[XLEN]    HVX_ALIGN;
static int8_t  taps_buf[TLEN]  HVX_ALIGN;
static int32_t bias_buf[C]     HVX_ALIGN;
static int8_t  out_buf[OLEN]   HVX_ALIGN;
static int8_t  ref_buf[OLEN]   HVX_ALIGN;

int main(void) {
    uint32_t s = 0xA1B2C3u;
    for (int i = 0; i < XLEN;  i++) x_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < TLEN;  j++) taps_buf[j]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int c = 0; c < C;     c++) bias_buf[c]  = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    x_buf[0]          = -128; taps_buf[0]        = -128;  /* ch0: max product */
    x_buf[XSTRIDE]    = -128; taps_buf[NTAPS]    = -128;  /* ch1: max product */
    bias_buf[0]       = -200000;   /* forces relu clamp for ch0 */
    bias_buf[1]       =  200000;   /* large positive bias, stays positive */
    x_buf[XLEN - 1]  = 127;       /* tail edge of last channel */

    /* Golden reference: depthwise 1D FIR + bias + relu + saturate */
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x_buf    + ch * XSTRIDE;
        const int8_t *tapch = taps_buf + ch * NTAPS;
        int8_t       *rch   = ref_buf  + ch * L;
        int32_t       b     = bias_buf[ch];
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < NTAPS; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            int32_t biased = acc + b;
            if (biased < 0) biased = 0;   /* relu */
            /* saturate int32 to int8 (values > 127 clamp to 127) */
            if (biased >  127) biased =  127;
            rch[i] = (int8_t)biased;
        }
    }

    /* Poison output */
    for (int i = 0; i < OLEN; i++) out_buf[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, taps_buf, bias_buf, out_buf, L, C, NTAPS); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < OLEN; i++) {
        if (out_buf[i] != ref_buf[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)out_buf[i]; expv = (long)ref_buf[i]; }
        }
    }
    hvx_report(errors, OLEN, fb, gotv, expv);
    return errors ? 1 : 0;
}
