#include "harness_common.h"
#include "kernel_api.h"
#define H_DIM 10
#define W_DIM 13
#define P_DIM 3
#define OH_DIM (H_DIM + 2*P_DIM)   /* 16 */
#define OW_DIM (W_DIM + 2*P_DIM)   /* 19 */
#define IN_SIZE  (H_DIM * W_DIM)    /* 130 */
#define OUT_SIZE (OH_DIM * OW_DIM)  /* 304 */
static int8_t in[IN_SIZE]   HVX_ALIGN;
static int8_t out[OUT_SIZE] HVX_ALIGN;
static int8_t ref[OUT_SIZE] HVX_ALIGN;
int main(void) {
    uint32_t s = 0x82B3C4u;
    /* Non-zero fill so border vs interior is detectable */
    for (int i = 0; i < IN_SIZE; i++) {
        in[i] = (int8_t)(hvx_lcg(&s) >> 24);
        /* Ensure no accidental zeros in interior (set bit 7 if zero) */
        if (in[i] == 0) in[i] = (int8_t)0x7F;
    }
    /* Golden reference */
    for (int r = 0; r < OH_DIM; r++) {
        for (int c = 0; c < OW_DIM; c++) {
            int sr = r - P_DIM, sc = c - P_DIM;
            ref[r * OW_DIM + c] = (sr >= 0 && sr < H_DIM && sc >= 0 && sc < W_DIM)
                                  ? in[sr * W_DIM + sc]
                                  : (int8_t)0;
        }
    }
    for (int i = 0; i < OUT_SIZE; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, H_DIM, W_DIM, P_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < OUT_SIZE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, OUT_SIZE, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
