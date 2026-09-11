#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int8_t in[N]   HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN, lut[256] HVX_ALIGN;

/* lo/hi pairs to sweep so candidate cannot hardcode the clamp params */
static const uint8_t LOS[] = { 32,  0,  64 };
static const uint8_t HIS[] = { 224, 127, 192 };
#define NSETS ((int)(sizeof(LOS)/sizeof(LOS[0])))

int main(void){
    uint32_t s = 0xD4C0FEu;
    /* single shared table across all sweeps (only lo/hi changes) */
    for (int k = 0; k < 256; k++) lut[k] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N; i++)   in[i]  = (int8_t)(hvx_lcg(&s) >> 24);

    int total_errors = 0, fb_global = -1;
    long gotv = 0, expv = 0;

    for (int t = 0; t < NSETS; t++){
        uint8_t lo = LOS[t], hi = HIS[t];
        /* inject boundary values into shared in[] */
        in[0] = (int8_t)(uint8_t)lo;
        in[1] = (int8_t)(uint8_t)hi;
        in[2] = (int8_t)(uint8_t)(lo > 0   ? lo - 1  : 0);
        in[3] = (int8_t)(uint8_t)(hi < 255 ? hi + 1  : 255);
        in[4] = (int8_t)0u;
        in[5] = (int8_t)255u;
        /* reference */
        for (int i = 0; i < N; i++){
            uint8_t idx = (uint8_t)in[i];
            if (idx < lo) idx = lo;
            if (idx > hi) idx = hi;
            ref[i] = lut[idx];
        }
        for (int i = 0; i < N; i++) out[i] = 0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, lut, lo, hi); });
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
