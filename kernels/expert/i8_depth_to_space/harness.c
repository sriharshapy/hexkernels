#include "harness_common.h"
#include "kernel_api.h"
#define C_IN  12
#define H_IN   4
#define W_IN   5
#define B_DIM  2
#define C_OUT (C_IN / (B_DIM * B_DIM))   /* 3 */
#define H_OUT (H_IN * B_DIM)              /* 8 */
#define W_OUT (W_IN * B_DIM)              /* 10 */
#define IN_SIZE  (C_IN  * H_IN  * W_IN)   /* 240 */
#define OUT_SIZE (C_OUT * H_OUT * W_OUT)   /* 240 */
static int8_t in[IN_SIZE]   HVX_ALIGN;
static int8_t out[OUT_SIZE] HVX_ALIGN;
static int8_t ref[OUT_SIZE] HVX_ALIGN;
int main(void) {
    uint32_t s = 0x71A2B3u;
    for (int i = 0; i < IN_SIZE; i++) in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference */
    for (int c = 0; c < C_OUT; c++)
        for (int bh = 0; bh < B_DIM; bh++)
            for (int bw = 0; bw < B_DIM; bw++) {
                int c_in = c*B_DIM*B_DIM + bh*B_DIM + bw;
                for (int oh = 0; oh < H_IN; oh++)
                    for (int ow = 0; ow < W_IN; ow++)
                        ref[c*H_OUT*W_OUT + (oh*B_DIM + bh)*W_OUT + (ow*B_DIM + bw)] =
                            in[c_in * H_IN*W_IN + oh*W_IN + ow];
            }
    for (int i = 0; i < OUT_SIZE; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, C_IN, H_IN, W_IN, B_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < OUT_SIZE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, OUT_SIZE, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
