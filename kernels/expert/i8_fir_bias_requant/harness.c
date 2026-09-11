#include "harness_common.h"
#include "kernel_api.h"

/* n=512, ntaps=16. VALID FIR: xlen = n+ntaps-1 = 527.
 * n=512 is a power of 2 (exercises aligned path too), ntaps=16 = one HVX vector width
 * in bytes (exercises full-width tap processing).
 * Sweep 3 (mult,shift,zp) sets + one bias value to prevent hardcoding. */
#define N     512
#define NTAPS 16
#define XLEN  (N + NTAPS - 1)  /* 527 */

static int8_t  x_buf[XLEN]     HVX_ALIGN;
static int8_t  taps_buf[NTAPS]  HVX_ALIGN;
static int8_t  out_buf[N]       HVX_ALIGN;
static int8_t  ref_buf[N]       HVX_ALIGN;

static const int32_t MULTS[]  = {  7,  1, -2 };
static const int     SHIFTS[] = {  8,  0,  5 };
static const int8_t  ZPS[]    = {  0, 10, -3 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

/* Use a non-trivial bias to ensure it's not ignored */
#define BIAS_VAL  500

static int8_t requant(int32_t biased, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)biased * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

int main(void) {
    uint32_t s = 0xF1D2A3u;
    for (int i = 0; i < XLEN;  i++) x_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < NTAPS; j++) taps_buf[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Max-magnitude products */
    x_buf[0]      = -128; taps_buf[0]  = -128;
    x_buf[1]      =  127; taps_buf[1]  = -128;
    x_buf[XLEN-1] = -128;

    int32_t bias = BIAS_VAL;

    int total_errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t mult  = MULTS[p];
        int     shift = SHIFTS[p];
        int8_t  zp    = ZPS[p];

        /* Build reference */
        for (int i = 0; i < N; i++) {
            int32_t acc = 0;
            for (int k = 0; k < NTAPS; k++)
                acc += (int32_t)x_buf[i + k] * (int32_t)taps_buf[k];
            ref_buf[i] = requant(acc + bias, mult, shift, zp);
        }

        /* Poison */
        for (int i = 0; i < N; i++) out_buf[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, taps_buf, bias, out_buf, N, NTAPS, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < N; i++) {
            if (out_buf[i] != ref_buf[i]) {
                total_errors++;
                if (fb < 0) { fb = p * N + i; gotv = (long)(int8_t)out_buf[i]; expv = (long)(int8_t)ref_buf[i]; }
            }
        }
    }
    hvx_report(total_errors, N * NSETS, fb, gotv, expv);
    return total_errors ? 1 : 0;
}
