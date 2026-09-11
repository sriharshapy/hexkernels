#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int8_t  a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int8_t ref_element(int8_t ai, int8_t bi,
                           int32_t mult, int shift, int8_t zp) {
    int32_t sum  = (int32_t)ai + (int32_t)bi;
    if (sum < 0) sum = 0;   /* ReLU */
    int64_t v    = (int64_t)sum * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep multiple (mult, shift, zp) sets; one negative mult ensures relu bites.
 * Inputs span [-80, 80] so many sums are negative and relu clamps them. */
static const int32_t MULTS[]  = {  5,  1, 11, -2, 63 };
static const int     SHIFTS[] = {  4,  0,  6,  3,  6 };
static const int     ZPS[]    = {  0,  0, -3,  8,  0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xFEEDCAFu;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)((int32_t)(hvx_lcg(&s)) % 161 - 80);
        b[i] = (int8_t)((int32_t)(hvx_lcg(&s)) % 161 - 80);
    }
    /* Edge cases for relu */
    a[0] =   60; b[0] =   60;  /* positive sum, no relu */
    a[1] =  -60; b[1] =  -60;  /* negative sum -> relu -> 0 */
    a[2] =   60; b[2] =  -61;  /* sum = -1 -> relu -> 0 */
    a[3] =   60; b[3] =  -60;  /* sum = 0 -> relu = 0 */
    a[4] =  127; b[4] =    0;
    a[5] = -128; b[5] =    0;  /* sum negative -> relu -> 0 */
    a[6] =  -1;  b[6] =   1;   /* sum = 0 */

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
