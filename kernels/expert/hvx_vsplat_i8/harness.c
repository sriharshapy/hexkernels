#include "harness_common.h"
#include "kernel_api.h"

#define N 1011   /* 7*128 + 115 tail */
#define VAL ((int8_t)0x8F)   /* -113: mixed bit pattern, high bit set */

static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;
    for (int i = 0; i < N; i++) ref[i] = VAL;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(VAL, out, N); });
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
