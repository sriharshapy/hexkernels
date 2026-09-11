#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define K 32
#define NCOL 64

static uint8_t A[M*K]     HVX_ALIGN;
static int8_t  B[K*NCOL]  HVX_ALIGN;
static int32_t out[M*NCOL] HVX_ALIGN;
static int32_t ref[M*NCOL] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x3EE1u;
    for (int i = 0; i < M*K;    i++) A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < K*NCOL; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < NCOL; j++) {
            int acc = 0;
            for (int k = 0; k < K; k++) acc += (int)A[i*K+k] * (int)B[k*NCOL+j];
            ref[i*NCOL+j] = sx12((acc * 17 + 8) >> 4);
        }

    for (int i = 0; i < M*NCOL; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, M, NCOL, K); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < M*NCOL; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, M*NCOL, fb, gotv, expv);
    return errors ? 1 : 0;
}
