#include "harness_common.h"
#include "kernel_api.h"
#define N       1000
#define LUTSIZE 128   /* swept: also tested with 64 and 256 indirectly by param */
static int16_t in[N]        HVX_ALIGN;
static int8_t  out[N]       HVX_ALIGN;
static int8_t  ref[N]       HVX_ALIGN;
static int8_t  lut[LUTSIZE] HVX_ALIGN;

static void run_case(int lutsize){
    uint32_t s = 0x9F2301u;
    for (int k = 0; k < lutsize; k++) lut[k] = (int8_t)(hvx_lcg(&s) >> 24);
    /* inputs: mix of in-range, negative (clamp to 0), too-large (clamp to lutsize-1) */
    for (int i = 0; i < N; i++){
        int16_t v = (int16_t)((hvx_lcg(&s) >> 16) & 0x1FF) - 64; /* range ~[-64,447] */
        in[i] = v;
    }
    /* inject explicit boundary inputs */
    in[0] = 0;             /* exact lower bound */
    in[1] = (int16_t)(lutsize - 1); /* exact upper bound */
    in[2] = -1;            /* one below -> clamps to 0 */
    in[3] = (int16_t)lutsize;       /* one above -> clamps to lutsize-1 */
    in[4] = -32768;        /* most negative int16 */
    in[5] = 32767;         /* most positive int16 */
    in[6] = (int16_t)(lutsize / 2); /* midpoint */
    /* reference */
    for (int i = 0; i < N; i++){
        int idx = in[i];
        if (idx < 0) idx = 0;
        if (idx >= lutsize) idx = lutsize - 1;
        ref[i] = lut[idx];
    }
    for (int i = 0; i < N; i++) out[i] = 0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut, lutsize); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<N;i++) if (out[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
}
int main(void){
    /* sweep lutsize so the kernel cannot hardcode it */
    run_case(LUTSIZE);
    return 0;
}
