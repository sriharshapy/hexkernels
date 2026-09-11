#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
/* lut has 17 entries: lut[0]..lut[16] so lut[hi+1] is valid when hi=15. */
static int8_t in[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN, lut[17] HVX_ALIGN;
int main(void){
    uint32_t s = 0xA1B2C3u;
    for (int i = 0; i < N; i++)    in[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < 17; k++)   lut[k] = (int8_t)(hvx_lcg(&s) >> 24);
    /* inject boundary values: lo=0 (exact), lo=15 (almost next), hi=0, hi=15 */
    in[0] = (int8_t)0x00u; /* hi=0, lo=0 */
    in[1] = (int8_t)0x0Fu; /* hi=0, lo=15 */
    in[2] = (int8_t)0xF0u; /* hi=15, lo=0 */
    in[3] = (int8_t)0xFFu; /* hi=15, lo=15 */
    in[4] = (int8_t)0x80u; /* hi=8, lo=0, tests sign bit */
    /* compute reference */
    for (int i = 0; i < N; i++){
        uint8_t b  = (uint8_t)in[i];
        int     hi = b >> 4;
        int     lo = b & 0xF;
        int16_t base  = lut[hi];
        int16_t delta = (int16_t)(lut[hi + 1] - lut[hi]);
        ref[i] = (int8_t)(base + ((delta * lo) >> 4));
    }
    for (int i = 0; i < N; i++) out[i] = 0xA5;             /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<N;i++) if (out[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
