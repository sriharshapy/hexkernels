#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int32_t acc[N] HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int8_t requant_ref(int32_t ai, int32_t mult, int shift) {
    int64_t v = (int64_t)ai * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    if (r > 127)  r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep (mult, shift): the two required cases -- a real shift with rounding,
 * and shift=0 (half=0, no-rounding branch). */
static const int32_t MULTS[]  = { 200,   1 };
static const int     SHIFTS[] = {   8,   0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xF00DCAFEu;
    /* |acc| <= 10,000,000 so acc*mult (mult<=200) stays within int32 range by
     * construction -- see kernel_api.h for why (no native HVX 32x32->64 widen). */
    for (int i = 0; i < N; i++)
        acc[i] = (int32_t)((int32_t)(hvx_lcg(&s) % 20000001) - 10000000);

    acc[0] = 10000000;    /* large positive -- near the safe-bound ceiling, saturates both sets */
    acc[1] = -10000000;   /* large negative -- saturates both sets (-128) */
    acc[2] = 0;           /* zero */
    acc[3] = 1;           /* small positive */
    acc[4] = 16;          /* rounding TIE for (mult=200,shift=8): v=3200, exactly x.5 * 256 */
    acc[5] = -16;         /* symmetric negative tie */
    acc[6] = -1;          /* small negative */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int kk = 0; kk < NSETS; kk++) {
        int32_t mult = MULTS[kk]; int shift = SHIFTS[kk];
        for (int i = 0; i < N; i++) ref[i] = requant_ref(acc[i], mult, shift);
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;   /* poison */

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(acc, out, N, mult, shift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < N; i++)
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = kk * N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
    }
    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
