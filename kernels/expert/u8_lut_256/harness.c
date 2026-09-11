#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static uint8_t in[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN, lut[256] HVX_ALIGN;
int main(void){
    uint32_t s = 0x5E3A19u;
    for (int i = 0; i < N; i++)   in[i]  = (uint8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < 256; k++) lut[k] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* inject index boundary values */
    in[0]=0; in[1]=255; in[2]=1; in[3]=254; in[4]=127; in[5]=128;
    for (int i = 0; i < N; i++) ref[i] = lut[in[i]];
    for (int i = 0; i < N; i++) out[i] = 0xA5;             /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<N;i++) if (out[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
