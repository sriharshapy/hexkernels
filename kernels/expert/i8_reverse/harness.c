#include "harness_common.h"
#include "kernel_api.h"
#define N 501
static int8_t in[N] HVX_ALIGN, out[N] HVX_ALIGN, ref[N] HVX_ALIGN;
int main(void) {
    uint32_t s = 0xA4D5E6u;
    for (int i = 0; i < N; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference */
    for (int i = 0; i < N; i++) ref[i] = in[N - 1 - i];
    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, N, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
