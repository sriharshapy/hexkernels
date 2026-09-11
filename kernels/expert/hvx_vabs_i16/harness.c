#include "harness_common.h"
#include "kernel_api.h"

#define N 513   /* 8*64 + 1 tail (64 int16 lanes/vector) */

static int16_t a[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xAB51u;
    for (int i = 0; i < N; i++)
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);

    /* Pinned edge cases. */
    a[0] = -32768;  /* WRAP corner: abs(-32768) = -32768, not 32767/32768 */
    a[1] = -1;      /* abs(-1) = 1 */
    a[2] = 5;       /* abs(5) = 5 (already positive) */
    a[3] = -32767;  /* abs(-32767) = 32767 (max representable, no wrap) */
    /* Tail-path edge: last element (index N-1 = 512, the single-element tail). */
    a[N-1] = -42;

    for (int i = 0; i < N; i++) {
        int v = a[i];
        ref[i] = (v == -32768) ? (int16_t)-32768 : (int16_t)(v < 0 ? -v : v);
    }

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N); });
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
