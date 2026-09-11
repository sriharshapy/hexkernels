#include "harness_common.h"
#include "kernel_api.h"
#define M 64
#define K 128

/* acc max |val| = 128*128*128 = 2,097,152 << INT32_MAX.
 * bias in [-32768, 32767]. */
static int8_t  A[M*K]    HVX_ALIGN;
static int8_t  x[K]      HVX_ALIGN;
static int32_t bias[M]   HVX_ALIGN;
static int8_t  out[M]    HVX_ALIGN;
static int8_t  ref[M]    HVX_ALIGN;

static int8_t requant_ref(int32_t acc, int32_t b, int32_t mult, int shift, int8_t zp) {
    int64_t biased = (int64_t)acc + (int64_t)b;
    int64_t v    = biased * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static const int32_t MULTS[]  = {  3,  1, 15, -2,  63 };
static const int     SHIFTS[] = {  4,  0,  7,  3,   5 };
static const int     ZPS[]    = {  0,  0, -5, 10,   0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xE1F7C3u;
    for (int i = 0; i < M*K; i++) A[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int k = 0; k < K;   k++) x[k]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < M;   i++) bias[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases: large products, sign in reduction, bias edge cases */
    A[0] = 127;  x[0] = 127;           /* large positive */
    A[K] = -128; x[1] = 127;           /* sign edge in row 1 */
    bias[0] = -2000000;                /* large negative bias */
    bias[1] =  2000000;                /* large positive bias */
    bias[2] = 0;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int p = 0; p < NSETS; p++) {
        int32_t mult = MULTS[p]; int shift = SHIFTS[p]; int8_t zp = (int8_t)ZPS[p];

        /* Build reference */
        for (int i = 0; i < M; i++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)A[i*K+k] * (int32_t)x[k];
            ref[i] = requant_ref(acc, bias[i], mult, shift, zp);
        }

        /* Poison output */
        for (int i = 0; i < M; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, x, bias, out, M, K, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*M + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, M*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
