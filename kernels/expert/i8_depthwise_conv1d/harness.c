#include "harness_common.h"
#include "kernel_api.h"

/* C=4 channels, L=250 output samples/channel, ntaps=8.
   Total output = C*L = 1000. xlen per channel = L+ntaps-1 = 257.
   n=250 per channel is not a multiple of 128 — exercises tail. */
#define C     4
#define L     250
#define NTAPS 8
#define XSTRIDE (L + NTAPS - 1)   /* 257 samples per channel in x */
#define XLEN  (C * XSTRIDE)       /* 1028 total */
#define TLEN  (C * NTAPS)         /* 32 */
#define OLEN  (C * L)             /* 1000 */

static int8_t  x[XLEN]  HVX_ALIGN;
static int8_t  taps[TLEN] HVX_ALIGN;
static int32_t out[OLEN] HVX_ALIGN;
static int32_t ref[OLEN] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xD3E85Fu;
    for (int i = 0; i < XLEN; i++)  x[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < TLEN; j++)  taps[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Inject max-magnitude products across channels */
    x[0]             = -128; taps[0]         = -128;   /* ch0, pos 0 */
    x[XSTRIDE]       = -128; taps[NTAPS]     = -128;   /* ch1, pos 0 */
    x[XLEN - 1]      = -128;                            /* last sample (tail edge) */

    /* Golden reference */
    for (int ch = 0; ch < C; ch++) {
        const int8_t *xch   = x    + ch * XSTRIDE;
        const int8_t *tapch = taps + ch * NTAPS;
        int32_t      *rch   = ref  + ch * L;
        for (int i = 0; i < L; i++) {
            int32_t acc = 0;
            for (int j = 0; j < NTAPS; j++)
                acc += (int32_t)xch[i + j] * (int32_t)tapch[j];
            rch[i] = acc;
        }
    }

    /* Poison */
    for (int i = 0; i < OLEN; i++) out[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, taps, out, L, C, NTAPS); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < OLEN; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, OLEN, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
