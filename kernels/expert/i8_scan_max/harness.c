#include "harness_common.h"
#include "kernel_api.h"
#define N 1000

static int8_t in[N]  HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void){
    uint32_t s = 0x7E8F9A0Bu;
    for (int i=0; i<N; i++) in[i]=(int8_t)(hvx_lcg(&s)>>24);
    /* Edge cases */
    in[0]   = -128;  /* minimum at start: first output must be -128 */
    in[1]   =  127;  /* jump to global max at i=1: all subsequent >= 127 */
    in[500] = -128;  /* late minimum: should NOT drop the running max */
    in[999] = -128;  /* final tail: still should be the max seen so far */
    /* Monotone-decreasing run: running max stays at first element */
    for (int i=50; i<70; i++) in[i]=(int8_t)(50-i);
    /* Monotone-increasing run: running max tracks last element */
    for (int i=200; i<220; i++) in[i]=(int8_t)(i-200-100); /* -100..-81 then rise */
    /* Reset to all-max at 300 */
    in[300] = 127;

    /* Reference */
    int8_t cur=-128;
    for (int i=0; i<N; i++){
        if(in[i]>cur) cur=in[i];
        ref[i]=cur;
    }
    /* Poison */
    for (int i=0; i<N; i++) out[i]=(int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1; long gotv=0, expv=0;
    for (int i=0; i<N; i++){
        if(out[i]!=ref[i]){ errors++; if(fb<0){fb=i;gotv=(long)out[i];expv=(long)ref[i];} }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
