#include "harness_common.h"
#include "kernel_api.h"
#define N 32

static uint8_t A[N*N] HVX_ALIGN;
static int8_t  x[N]   HVX_ALIGN;
static int32_t out[N] HVX_ALIGN;
static int32_t ref[N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x6E71u;
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int k = 0; k < N;   k++) x[k] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int i = 0; i < N; i++) {
        int acc = 0;
        for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)x[k];
        ref[i] = sx12((acc * 17 + 8) >> 4);
    }

    for (int i = 0; i < N; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, x, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
