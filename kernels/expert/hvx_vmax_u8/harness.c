#include "harness_common.h"
#include "kernel_api.h"

#define N 1013   /* 7*128 + 117 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00CA00u;
    for (int i = 0; i < N; i++) {
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    a[0] = 0;   b[0] = 255;   /* max=255 */
    a[1] = 128; b[1] = 127;   /* unsigned max=128 (signed compare would say 127) */
    a[2] = 255; b[2] = 255;   /* equal operands -> 255 */
    a[3] = 200; b[3] = 100;   /* max=200 */
    a[N-1] = 128; b[N-1] = 0; /* tail-path edge -> 128 */

    for (int i = 0; i < N; i++)
        ref[i] = (a[i] > b[i]) ? a[i] : b[i];

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
