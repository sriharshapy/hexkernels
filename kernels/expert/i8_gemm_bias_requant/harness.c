#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define N 32
#define K 32

/* acc max |val| = 128*128*48 = 786,432 << INT32_MAX.
 * bias up to 32767 (GEMM+bias still fits int32). */
static int8_t  A[M*K]    HVX_ALIGN;
static int8_t  B[K*N]    HVX_ALIGN;
static int32_t bias[N]   HVX_ALIGN;
static int8_t  out[M*N]  HVX_ALIGN;
static int8_t  ref[M*N]  HVX_ALIGN;

static int8_t requant_bias_ref(int32_t acc, int32_t b, int32_t mult, int shift, int8_t zp) {
    int64_t biased = (int64_t)acc + (int64_t)b;
    /* NO relu -- negatives flow through to requant */
    int64_t v    = biased * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep to prevent hardcoding quant params. */
static const int32_t MULTS[]  = { 3, 1 };
static const int     SHIFTS[] = { 4, 0 };
static const int     ZPS[]    = { 0, 0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xC9D4E2u;
    for (int i = 0; i < M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N;   i++) bias[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    A[0] = 127;  B[0] = 127;           /* large positive product */
    A[1] = -128; B[N] = 127;           /* sign edge */
    bias[0] = -2000000;                /* large negative bias: no relu so negatives survive */
    bias[1] =  2000000;                /* large positive bias: may saturate to 127 */
    bias[2] = 0;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        int32_t mult = MULTS[p]; int shift = SHIFTS[p]; int8_t zp = (int8_t)ZPS[p];

        /* Build reference */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[k*N+j];
                ref[i*N+j] = requant_bias_ref(acc, bias[j], mult, shift, zp);
            }
        }

        /* Poison output */
        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, out, M, N, K, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*M*N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, M*N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
