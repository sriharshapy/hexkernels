#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int32_t a[N]    HVX_ALIGN;
static int32_t bias[N] HVX_ALIGN;
static int8_t  out[N]  HVX_ALIGN, ref[N] HVX_ALIGN;

static int8_t bias_requant_ref(int32_t ax, int32_t bx, int32_t mult, int shift, int8_t zp) {
    int64_t sum = (int64_t)ax + (int64_t)bx;
    int64_t v   = sum * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r   = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r > 127)  r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Sweep (mult, shift, zp): negative mult, shift==0, non-zero zp. */
static const int32_t MULTS[]  = {  5,  1,  13, -3, 127 };
static const int     SHIFTS[] = {  3,  0,   7,  2,   4 };
static const int     ZPS[]    = {  0,  0,  -5, 10,   0 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

int main(void) {
    uint32_t s = 0xBEEF77u;
    for (int i = 0; i < N; i++) {
        a[i]    = (int32_t)((int32_t)(hvx_lcg(&s)) % 401) - 200;
        bias[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 201) - 100;
    }
    /* inject boundary values */
    a[0] = 200;  bias[0] = 100;   /* large positive sum */
    a[1] = -200; bias[1] = -100;  /* large negative sum */
    a[2] = 2;    bias[2] = 2;     /* rounding tie at mult=5,shift=3 */
    a[3] = 0;    bias[3] = 0;
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int k = 0; k < NSETS; k++) {
        int32_t mult = MULTS[k]; int shift = SHIFTS[k]; int8_t zp = (int8_t)ZPS[k];
        for (int i = 0; i < N; i++) ref[i] = bias_requant_ref(a[i], bias[i], mult, shift, zp);
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, bias, out, N, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++)
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = k * N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
    }
    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
