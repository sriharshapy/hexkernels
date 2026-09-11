#include "harness_common.h"
#include "kernel_api.h"
#define C_DIM 4
#define H_DIM 10
#define W_DIM 13
#define N_ELEM (C_DIM*H_DIM*W_DIM)
static int8_t in[N_ELEM] HVX_ALIGN, out[N_ELEM] HVX_ALIGN, ref[N_ELEM] HVX_ALIGN;
int main(void) {
    uint32_t s = 0x4E5F60u;
    for (int i = 0; i < N_ELEM; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference: out[c*H*W + h*W + w] = in[h*W*C + w*C + c] */
    for (int h = 0; h < H_DIM; h++)
        for (int w = 0; w < W_DIM; w++)
            for (int c = 0; c < C_DIM; c++)
                ref[c*H_DIM*W_DIM + h*W_DIM + w] = in[h*W_DIM*C_DIM + w*C_DIM + c];
    for (int i = 0; i < N_ELEM; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, C_DIM, H_DIM, W_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < N_ELEM; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, N_ELEM, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
