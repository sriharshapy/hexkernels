#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int32_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int8_t ref_element(int32_t ai, int32_t bi, int32_t mult, int shift, int8_t zp) {
    /* Step 1: add (int32, wraps on overflow -- use int64 for safety) */
    int64_t sum = (int64_t)ai + (int64_t)bi;
    /* Step 2: relu -- clamp to [0, INT32_MAX] */
    if (sum < 0) sum = 0;
    /* Step 3: requantize round-half-away-from-zero */
    int64_t v    = sum * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep multiple (mult, shift, zp) sets so a candidate MUST read params at runtime.
 * Includes negative mult (sign handling), shift==0, non-zero zp, large positive,
 * and a set where relu really bites (many negative sums clamped to 0). */
static const int32_t MULTS[]  = {  5,  1, 13, -3, 127 };
static const int     SHIFTS[] = {  4,  0,  7,  2,   5 };
static const int     ZPS[]    = {  0,  0, -5, 10,   0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xDEAD77u;
    /* Random inputs in [-600, 600] so many sums are negative -> relu matters */
    for (int i = 0; i < N; i++) {
        a[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 601) - 300;
        b[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 601) - 300;
    }
    /* Injected edge cases */
    a[0] =  300; b[0] =  300;   /* large positive, no relu */
    a[1] = -300; b[1] = -300;   /* relu clamps to 0 */
    a[2] =  300; b[2] = -301;   /* just negative -> relu -> 0 */
    a[3] =  300; b[3] = -300;   /* sum=0 -> relu=0 */
    a[4] = 0x7FFFFFFF; b[4] = 0; /* large positive */
    a[5] = 0; b[5] = 0;

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
                if (fb < 0) { fb = k*N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, N*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
