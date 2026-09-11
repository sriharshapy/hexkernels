#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static uint8_t in[N]   HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN, lut[256] HVX_ALIGN;

/* sweep k so candidate cannot hardcode it */
static const int KS[] = { 3, 7, 2, 5 };
#define NSETS ((int)(sizeof(KS)/sizeof(KS[0])))

int main(void){
    uint32_t s = 0xF1E2D3u;
    for (int i = 0; i < N; i++)   in[i]  = (uint8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < 256; k++) lut[k] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* inject boundary index values at positions divisible by each k */
    in[0]   = 0;   /* lut[0] for k=2,3,5,7 (i=0 always divisible) */
    in[7]   = 255; /* lut[255] for k=7 */
    in[6]   = 128; /* lut[128] for k=2,3 */
    in[1]   = 127; /* copy for k=3,5,7; lut for k=2 */

    int total_errors = 0, fb_global = -1;
    long gotv = 0, expv = 0;
    for (int t = 0; t < NSETS; t++){
        int stride = KS[t];
        for (int i = 0; i < N; i++)
            ref[i] = (i % stride == 0) ? lut[in[i]] : in[i];
        for (int i = 0; i < N; i++) out[i] = 0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut, stride); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++){
            if (out[i] != ref[i]){
                total_errors++;
                if (fb_global < 0){ fb_global = t*N+i; gotv=(long)out[i]; expv=(long)ref[i]; }
            }
        }
    }
    hvx_report(total_errors, N*NSETS, fb_global, gotv, expv);
    return total_errors ? 1 : 0;
}
