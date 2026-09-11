#include "harness_common.h"
#include "kernel_api.h"
#define N 32
#define KK 64

static uint8_t A[N*KK]  HVX_ALIGN;
static int8_t  B[KK*N]  HVX_ALIGN;
static int32_t out[N*N] HVX_ALIGN;
static int32_t ref[N*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x4E27u;
    /* A 0..7, B -3..3, K=64 -> |acc| <= 64*7*3 = 1344 -> |r| <= 1428 < 2048. */
    for (int i = 0; i < N*KK; i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < KK*N; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int acc = 0;
            for (int k = 0; k < KK; k++) acc += (int)A[i*KK+k] * (int)B[k*N+j];
            ref[i*N+j] = sx12((acc * 17 + 8) >> 4);
        }

    for (int i = 0; i < N*N; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, N, KK); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
