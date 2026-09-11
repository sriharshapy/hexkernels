#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int8_t in[N]      HVX_ALIGN;
static int8_t out[N]     HVX_ALIGN;
static int8_t ref[N]     HVX_ALIGN;
static int8_t tableA[128] HVX_ALIGN;  /* for negative inputs: -128..-1 */
static int8_t tableB[128] HVX_ALIGN;  /* for non-negative inputs: 0..127 */
int main(void){
    uint32_t s = 0x7C4E11u;
    for (int i = 0; i < N; i++)     in[i]     = (int8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < 128; k++)   tableA[k] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < 128; k++)   tableB[k] = (int8_t)(hvx_lcg(&s) >> 24);
    /* inject boundary values */
    in[0] = -128; /* tableA[0] */
    in[1] = -1;   /* tableA[127] */
    in[2] =  0;   /* tableB[0] */
    in[3] =  127; /* tableB[127] */
    in[4] = -64;  /* tableA[64] */
    in[5] =  64;  /* tableB[64] */
    /* reference */
    for (int i = 0; i < N; i++){
        if (in[i] < 0)
            ref[i] = tableA[(uint8_t)in[i] - 128];
        else
            ref[i] = tableB[in[i]];
    }
    for (int i = 0; i < N; i++) out[i] = 0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, tableA, tableB); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<N;i++) if (out[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
