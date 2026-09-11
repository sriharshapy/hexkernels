#include "harness_common.h"
#include "kernel_api.h"

#define N 1069   /* 8*128 + 45 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00AD1Fu;
    for (int i = 0; i < N; i++) {
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    a[0] = 0;   b[0] = 255;   /* |0-255|=255 (max magnitude) */
    a[1] = 255; b[1] = 0;     /* |255-0|=255 (order reversed) */
    a[2] = 100; b[2] = 100;   /* equal -> 0 */
    a[3] = 128; b[3] = 127;   /* |128-127|=1 */
    a[N-1] = 10; b[N-1] = 200; /* tail-path edge: |10-200|=190 */

    for (int i = 0; i < N; i++) {
        int d = (int)a[i] - (int)b[i];
        ref[i] = (uint8_t)(d < 0 ? -d : d);
    }

    for (int i = 0; i < N; i++) out[i] = (uint8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
