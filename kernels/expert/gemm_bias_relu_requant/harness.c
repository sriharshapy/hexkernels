#include "harness_common.h"
#include "kernel_api.h"
#define M 32
#define N 32
#define K 32

/* Inputs bounded: |A|,|B| <= 127; |acc| <= 127*127*64 = 1,032,256 << INT32_MAX.
 * bias in [-200000, 200000] -- within int32 range after adding to acc. */
static int8_t  A[M*K]    HVX_ALIGN;
static int8_t  B[K*N]    HVX_ALIGN;
static int32_t bias[N]   HVX_ALIGN;
static int8_t  out[M*N]  HVX_ALIGN;
static int8_t  ref[M*N]  HVX_ALIGN;

static int8_t ref_element(int32_t acc, int32_t b, int32_t mult, int shift, int8_t zp) {
    /* add bias */
    int64_t biased = (int64_t)acc + (int64_t)b;
    /* relu */
    if (biased < 0) biased = 0;
    /* requantize round-half-away-from-zero */
    int64_t v    = biased * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep 2 (mult, shift, zp) parameter sets to prevent hardcoding.
 * Same bias array reused across sweeps (bias is input data, not a quant param).
 * 2 sets is sufficient: nearmiss_hardcode (mult=3,shift=4,zp=0) passes set 0 but fails set 1. */
static const int32_t MULTS[]  = { 3, 15 };
static const int     SHIFTS[] = { 4, 7 };
static const int     ZPS[]    = { 0, -5 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xF1C2B3u;
    for (int i = 0; i < M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*N; i++) B[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < N;   i++) bias[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    /* Edge cases: large products, sign in reduction, zero bias, big bias that flips relu */
    A[0] = 127;  B[0] = 127;
    A[1] = -128; B[N] = 127;
    bias[0] = -2000000;   /* large negative bias: acc + bias < 0 so relu kicks in */
    bias[1] = 2000000;    /* large positive bias: large output after requant */
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
                ref[i*N+j] = ref_element(acc, bias[j], mult, shift, zp);
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
