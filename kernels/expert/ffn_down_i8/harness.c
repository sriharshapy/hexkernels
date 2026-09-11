#include "harness_common.h"
#include "kernel_api.h"

#define M 8
#define N 8
#define K 110   /* NOT a multiple of 4: 110 = 27*4 + 2 (2-element tail) */

/* acc max |val| = 127*127*110 = 1,774,190 -- easily within int32. */
static int8_t  A[M*K]   HVX_ALIGN;
static int8_t  B[N*K]   HVX_ALIGN;   /* B TRANSPOSED: row j contiguous over K */
static int32_t bias[N]  HVX_ALIGN;
static int8_t  out[M*N] HVX_ALIGN;
static int8_t  ref[M*N] HVX_ALIGN;

static int8_t requant_ref(int32_t acc, int32_t b, int32_t mult, int shift) {
    int64_t biased = (int64_t)acc + (int64_t)b;
    int64_t r    = biased * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -(((-r) + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

/* Sweep to prevent hardcoding quant params. */
static const int32_t SMULTS[]  = { 3, 5 };
static const int     SSHIFTS[] = { 4, 6 };
#define NSETS ((int)(sizeof(SMULTS)/sizeof(SMULTS[0])))

int main(void) {
    uint32_t s = 0xB4A9C71u;
    for (int i = 0; i < M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N*K; i++) B[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    A[0*K+0] = 127; B[0*K+0] = 127;          /* large positive product at k=0 */
    A[1*K+0] = -128; B[1*K+0] = 127;         /* sign edge in reduction */
    /* K tail: nonzero values in the tail group [108,109] (110 = 27*4 + 2) */
    for (int i = 0; i < M; i++) { A[i*K+108] = 100; A[i*K+109] = -50; }
    for (int j = 0; j < N; j++) { B[j*K+108] = -80; B[j*K+109] = 60; }
    /* Negative and large bias values -- no ReLU here so negatives flow through */
    bias[0] = -500000;
    bias[1] =  500000;
    bias[2] = 0;
    bias[3] = -17;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t mult = SMULTS[p];
        int     shift = SSHIFTS[p];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)B[j*K+k];
                ref[i*N+j] = requant_ref(acc, bias[j], mult, shift);
            }
        }

        for (int i = 0; i < M*N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, B, bias, out, M, N, K, mult, shift); });
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
