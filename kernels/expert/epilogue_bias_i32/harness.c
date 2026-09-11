#include "harness_common.h"
#include "kernel_api.h"
#define R 20
#define C 100
static int32_t acc[R*C] HVX_ALIGN;
static int32_t bias[C]  HVX_ALIGN;
static int32_t out[R*C] HVX_ALIGN, ref[R*C] HVX_ALIGN;

int main(void){
    uint32_t s = 0x77AACCu;
    for (int i=0;i<R*C;i++) acc[i]  = (int32_t)((int32_t)(hvx_lcg(&s) % 2001) - 1000);
    for (int c=0;c<C;c++)   bias[c] = (int32_t)((int32_t)(hvx_lcg(&s) % 2001) - 1000);

    /* Row0's acc is all zero -- output must equal bias exactly (verifies bias
     * isn't accidentally scaled/duplicated). */
    for (int c=0;c<C;c++) acc[0*C+c] = 0;

    /* Bias sign/magnitude mix + extremes (kept well within int32 headroom
     * together with the acc extremes below: max |sum| = 2,000,000,000 +
     * 1,000,000 = 2,001,000,000 < INT32_MAX). */
    bias[0] = 1000000; bias[1] = -1000000; bias[2] = 0;

    /* acc extremes on a different row, paired with the extreme bias columns. */
    acc[1*C+0] = 2000000000;
    acc[1*C+1] = -2000000000;

    for (int r=0;r<R;r++) for (int c=0;c<C;c++) ref[r*C+c] = acc[r*C+c] + bias[c];
    for (int i=0;i<R*C;i++) out[i] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(acc, bias, out, R, C); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1;
    for (int i=0;i<R*C;i++) if (out[i]!=ref[i]) { errors++; if (fb<0) fb=i; }
    hvx_report(errors, R*C, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
