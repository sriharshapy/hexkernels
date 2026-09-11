#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int32_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int8_t ref_element(int32_t ai, int32_t bi,
                           int32_t mult, int shift, int8_t zp) {
    int64_t sum  = (int64_t)ai + (int64_t)bi;
    int64_t v    = sum * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep multiple (mult, shift, zp) sets -- a candidate MUST read params at runtime.
 * Includes negative mult (sign handling), shift==0, non-zero zp, and large values. */
static const int32_t MULTS[]  = {  5,  1, 13, -3, 127 };
static const int     SHIFTS[] = {  3,  0,  7,  2,   4 };
static const int     ZPS[]    = {  0,  0, -5, 10,   0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xC0FFEE7u;
    /* Residual-style: inputs in [-1024, 1023] so sums span signed int32 well */
    for (int i = 0; i < N; i++) {
        a[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 2048) - 1024;
        b[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 2048) - 1024;
    }
    /* Inject edge cases */
    a[0] =  1000; b[0] =  1000;   /* large positive sum */
    a[1] = -1000; b[1] = -1000;   /* large negative sum */
    a[2] =  1000; b[2] = -1000;   /* sum = 0 */
    a[3] =  2000000000; b[3] =  1000000000; /* sum > INT32_MAX, overflows if not widened */
    a[4] = -2000000000; b[4] = -1000000000; /* sum < INT32_MIN, overflows if not widened */
    a[5] = 4; b[5] = 0;           /* rounding tie at shift=3 with mult=5 */
    a[6] = 0; b[6] = 0;

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int k = 0; k < NSETS; k++) {
        int32_t mult = MULTS[k]; int shift = SHIFTS[k]; int8_t zp = (int8_t)ZPS[k];
        for (int i = 0; i < N; i++) ref[i] = ref_element(a[i], b[i], mult, shift, zp);
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;   /* poison */
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = k * N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
