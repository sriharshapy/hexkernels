#include "harness_common.h"
#include "kernel_api.h"
#define N 32

static uint8_t In[N*N]  HVX_ALIGN;
static int8_t  W[N*N]   HVX_ALIGN;
static int32_t bias[N]  HVX_ALIGN;
static int32_t out[N*N] HVX_ALIGN;
static int32_t ref[N*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x11C8u;
    for (int i = 0; i < N*N; i++) In[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < N*N; i++) W[i]  = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0;

    for (int p = 0; p < N; p++)
        for (int co = 0; co < N; co++) {
            int acc = 0;
            for (int ci = 0; ci < N; ci++) acc += (int)In[p*N+ci] * (int)W[co*N+ci];
            ref[p*N+co] = sx12((acc * 17 + 8) >> 4) + bias[co];
        }

    for (int i = 0; i < N*N; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(In, W, bias, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
