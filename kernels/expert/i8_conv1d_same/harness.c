#include "harness_common.h"
#include "kernel_api.h"

/* n=1000, ntaps=7 (odd, centered). pad_left=3, pad_right=3.
   n=1000 is not a multiple of 128 — exercises tail.
   Boundary outputs (first 3 and last 3) access zero-padded region. */
#define N     1000
#define NTAPS 7
#define PAD_LEFT  ((NTAPS - 1) / 2)  /* 3 */

static int8_t  x[N]     HVX_ALIGN;
static int8_t  taps[NTAPS] HVX_ALIGN;
static int32_t out[N]   HVX_ALIGN;
static int32_t ref[N]   HVX_ALIGN;

int main(void) {
    uint32_t s = 0xE5C7A2u;
    for (int i = 0; i < N; i++)      x[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < NTAPS; j++)  taps[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Max-magnitude products */
    x[0]     = -128; taps[0] = -128;
    x[N-1]   = -128; taps[NTAPS-1] = -128;
    /* Boundary injection: position 0 uses pad_left=3 zeros on the left */

    /* Golden reference */
    for (int i = 0; i < N; i++) {
        int32_t acc = 0;
        for (int j = 0; j < NTAPS; j++) {
            int xi = i - PAD_LEFT + j;
            if (xi >= 0 && xi < N)
                acc += (int32_t)x[xi] * (int32_t)taps[j];
        }
        ref[i] = acc;
    }

    /* Poison */
    for (int i = 0; i < N; i++) out[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, taps, out, N, NTAPS); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
