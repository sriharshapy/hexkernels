#include "harness_common.h"
#include "kernel_api.h"

#define N 270   /* 4*64 + 14 tail */

static int16_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[2*N] HVX_ALIGN, ref[2*N] HVX_ALIGN;

static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

int main(void) {
    uint32_t s = 0x00CA71u;
    for (int i = 0; i < N; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    /* Pinned saturation edge cases (chunk 0). */
    a[0] = 200;   /* saturates to 127 */
    b[0] = -200;  /* saturates to -128 */
    a[1] = -50;   /* no saturation */
    b[1] = 50;    /* no saturation */
    /* Tail-chunk edge case (chunk starts at base=256, tail m=14; N-1=269
     * is local j=13 within that tail chunk). */
    a[N-1] = 32767;  /* saturates to 127 */
    b[N-1] = -32768; /* saturates to -128 */

    /* Reference: chunked concatenation formula (matches HVX vpack). */
    {
        int base = 0;
        while (base < N) {
            int m = (N - base < 64) ? (N - base) : 64;
            for (int j = 0; j < m; j++) {
                ref[2*base + j]     = sat8(b[base + j]);
                ref[2*base + m + j] = sat8(a[base + j]);
            }
            base += m;
        }
    }

    for (int i = 0; i < 2*N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < 2*N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, 2*N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
