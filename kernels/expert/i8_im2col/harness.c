#include "harness_common.h"
#include "kernel_api.h"
#define C_DIM 3
#define H_DIM 8
#define W_DIM 9
#define K_DIM 3
#define S_DIM 1
#define OH_DIM ((H_DIM - K_DIM) / S_DIM + 1)  /* 6 */
#define OW_DIM ((W_DIM - K_DIM) / S_DIM + 1)  /* 7 */
#define IN_SIZE  (C_DIM * H_DIM * W_DIM)       /* 216 */
#define OUT_SIZE (C_DIM * K_DIM * K_DIM * OH_DIM * OW_DIM)  /* 27 * 42 = 1134 */
static int8_t in[IN_SIZE]   HVX_ALIGN;
static int8_t out[OUT_SIZE] HVX_ALIGN;
static int8_t ref[OUT_SIZE] HVX_ALIGN;
int main(void) {
    uint32_t s = 0x5F6071u;
    for (int i = 0; i < IN_SIZE; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference */
    for (int c = 0; c < C_DIM; c++)
        for (int kh = 0; kh < K_DIM; kh++)
            for (int kw = 0; kw < K_DIM; kw++) {
                int row = c*K_DIM*K_DIM + kh*K_DIM + kw;
                for (int oh = 0; oh < OH_DIM; oh++)
                    for (int ow = 0; ow < OW_DIM; ow++) {
                        int col = oh*OW_DIM + ow;
                        ref[row * OH_DIM*OW_DIM + col] =
                            in[c*H_DIM*W_DIM + (oh*S_DIM + kh)*W_DIM + (ow*S_DIM + kw)];
                    }
            }
    for (int i = 0; i < OUT_SIZE; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, C_DIM, H_DIM, W_DIM, K_DIM, S_DIM, OH_DIM, OW_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < OUT_SIZE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, OUT_SIZE, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
