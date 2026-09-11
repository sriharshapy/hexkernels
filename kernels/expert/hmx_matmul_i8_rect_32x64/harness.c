#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define NN 64
#define KK 32

static uint8_t A[M*KK]   HVX_ALIGN;
static int8_t  B[KK*NN]  HVX_ALIGN;
static int32_t out[M*NN] HVX_ALIGN;
static int32_t ref[M*NN] HVX_ALIGN;

static inline int sx12(int f) { int v = f & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }

int main(void) {
    uint32_t s = 0xE6BAu;
    /* A 0..7, B -3..3, K=32 -> |acc| <= 32*7*3 = 672 -> |r| <= 714 < 2048. */
    for (int i = 0; i < M*KK; i++)  A[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < KK*NN; i++) B[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < NN; j++) {
            int acc = 0;
            for (int k = 0; k < KK; k++) acc += (int)A[i*KK+k] * (int)B[k*NN+j];
            ref[i*NN+j] = sx12((acc * 17 + 8) >> 4);
        }

    for (int i = 0; i < M*NN; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, out, M, NN, KK); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < M*NN; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, M*NN, fb, gotv, expv);
    return errors ? 1 : 0;
}
