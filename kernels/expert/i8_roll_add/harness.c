#include "harness_common.h"
#include "kernel_api.h"
#define N 1024

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Sweep multiple k values so a candidate MUST read k at runtime. */
static const int KS[] = { 1, 128, 300, 1023 };
#define NK ((int)(sizeof(KS)/sizeof(KS[0])))

int main(void) {
    uint32_t s = 0x7E1A3Bu;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }
    /* Inject boundary values: wrap-around add at extremes */
    a[0] = 127; b[0] =   1;  /* wrapped add: -128 */
    a[1] = -128; b[1] = -1;  /* wrapped add: 127 */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int ki = 0; ki < NK; ki++) {
        int k = KS[ki];
        for (int i = 0; i < N; i++)
            ref[i] = (int8_t)(a[(i + k) % N] + b[i]);
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5; /* poison */
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N, k); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = ki*N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, N * NK, fb, gotv, expv);
    return errors ? 1 : 0;
}
