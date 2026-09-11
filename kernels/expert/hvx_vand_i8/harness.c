#include "harness_common.h"
#include "kernel_api.h"

#define N 1029   /* 8*128 + 5 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xA4Du;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases. */
    a[0] = (int8_t)0xFF; b[0] = (int8_t)0x0F;   /* 0xFF & 0x0F = 0x0F = 15 */
    a[1] = (int8_t)0xAA; b[1] = (int8_t)0x55;   /* 0xAA & 0x55 = 0x00 = 0 (disjoint bits) */
    a[2] = (int8_t)0x00; b[2] = (int8_t)0xFF;   /* 0 & anything = 0 */
    a[3] = -1;           b[3] = -1;             /* all-ones & all-ones = all-ones = -1 */
    /* Tail-path edge: last element (index N-1 = 1028, inside the 5-elem tail). */
    a[N-1] = (int8_t)0xF0; b[N-1] = (int8_t)0x3C;  /* 0xF0 & 0x3C = 0x30 = 48 */

    for (int i = 0; i < N; i++)
        ref[i] = (int8_t)((uint8_t)a[i] & (uint8_t)b[i]);

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

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
