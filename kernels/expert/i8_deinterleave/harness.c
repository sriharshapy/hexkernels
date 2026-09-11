#include "harness_common.h"
#include "kernel_api.h"
#define N 500
static int8_t in[2*N] HVX_ALIGN;
static int8_t a_out[N] HVX_ALIGN, b_out[N] HVX_ALIGN;
static int8_t a_ref[N] HVX_ALIGN, b_ref[N] HVX_ALIGN;
int main(void) {
    uint32_t s = 0x2C3D4Eu;
    /* Position-dependent interleaved fill */
    for (int i = 0; i < 2*N; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference */
    for (int i = 0; i < N; i++) {
        a_ref[i] = in[2*i];
        b_ref[i] = in[2*i + 1];
    }
    /* Poison outputs */
    for (int i = 0; i < N; i++) { a_out[i] = (int8_t)0xA5; b_out[i] = (int8_t)0xA5; }
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, a_out, b_out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (a_out[i] != a_ref[i]) { errors++; if (fb < 0) fb = i; }
        if (b_out[i] != b_ref[i]) { errors++; if (fb < 0) fb = N + i; }
    }
    hvx_report(errors, 2*N,
               fb, fb >= 0 ? (long)(fb < N ? a_out[fb] : b_out[fb - N]) : 0,
               fb >= 0 ? (long)(fb < N ? a_ref[fb] : b_ref[fb - N]) : 0);
    return errors ? 1 : 0;
}
