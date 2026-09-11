#include "harness_common.h"
#include "kernel_api.h"

#define N 1050   /* 8*128 + 26 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x7A79u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases: signed comparison, not unsigned-byte comparison. */
    a[0] = -1;   b[0] = 1;     /* SIGNED max=1; unsigned-byte max would wrongly report -1 (0xFF) */
    a[1] = -1;   b[1] = -128;  /* signed: -1 > -128 -> max=-1 */
    a[2] = 127;  b[2] = -128;  /* signed: 127 is max */
    a[3] = -50;  b[3] = -50;   /* equal -> either, result -50 */
    /* Tail-path edge: last element (index N-1 = 1049, inside the 26-elem tail). */
    a[N-1] = 3; b[N-1] = -3;

    for (int i = 0; i < N; i++)
        ref[i] = (a[i] > b[i]) ? a[i] : b[i];

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
