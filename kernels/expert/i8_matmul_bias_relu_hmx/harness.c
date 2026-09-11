#include "harness_common.h"
#include "kernel_api.h"
#define N 64

static uint8_t A[N*N]   HVX_ALIGN;
static int8_t  B[N*N]   HVX_ALIGN;
static int32_t bias[N]  HVX_ALIGN;
static int32_t out[N*N] HVX_ALIGN;
static int32_t ref[N*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x62B3u;
    /* Same matmul domain as i8_matmul_hmx_64x64: A in 0..7, B in -3..3, so
     * |acc|<=64*7*3=1344 -> |r|<=1428 (12-bit field never wraps). Bias spans
     * +-2000 so biased crosses zero often -> exercises both ReLU branches. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0; /* edge: forces-zero, forces-passthrough, zero bias */

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < N; k++) acc += (int)A[i*N+k] * (int)B[k*N+j];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[j];
            ref[i*N+j] = biased > 0 ? biased : 0;
        }

    for (int i = 0; i < N*N; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
