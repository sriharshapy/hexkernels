#include "harness_common.h"
#include "kernel_api.h"

/* stride=2 is the pinned param. n outputs, so x needs n*stride+ntaps-1 samples.
   n=501 (not a multiple of 128). ntaps=8, stride=2 → xlen=1010. */
#define N     501
#define NTAPS 8
#define STRIDE 2
#define XLEN  (N * STRIDE + NTAPS - 1)  /* 1010 */

static int8_t  x[XLEN]  HVX_ALIGN;
static int8_t  taps[NTAPS] HVX_ALIGN;
static int32_t out[N]   HVX_ALIGN;
static int32_t ref[N]   HVX_ALIGN;

int main(void) {
    uint32_t s = 0xB7C23Eu;
    for (int i = 0; i < XLEN; i++)  x[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < NTAPS; j++) taps[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Inject max-magnitude products and tail edges */
    x[0]       = -128; taps[0] = -128;   /* max product at i=0 */
    x[1]       =  127; taps[1] = -128;   /* second max */
    x[XLEN-1]  = -128;                   /* last sample exercised by tail */

    /* Golden reference (correlation, stride 2) */
    for (int i = 0; i < N; i++) {
        int32_t acc = 0;
        for (int j = 0; j < NTAPS; j++)
            acc += (int32_t)x[i * STRIDE + j] * (int32_t)taps[j];
        ref[i] = acc;
    }

    /* Poison output buffer */
    for (int i = 0; i < N; i++) out[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, taps, out, N, NTAPS, STRIDE); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
