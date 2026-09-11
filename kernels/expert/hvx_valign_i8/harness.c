#include "harness_common.h"
#include "kernel_api.h"

#define N 1024   /* 8 full 128B blocks, no tail */
#define RT 37

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xA119u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned known pattern in block 0: identity ramps for hand-verification
       (a[i]=100+i, b[i]=i for i in [0,128)). */
    for (int i = 0; i < 128; i++) { a[i] = (int8_t)(100 + i); b[i] = (int8_t)i; }

    for (int base = 0; base + 128 <= N; base += 128)
        for (int i = 0; i < 128; i++) {
            int idx = RT + i;
            ref[base + i] = (idx < 128) ? b[base + idx] : a[base + idx - 128];
        }

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
