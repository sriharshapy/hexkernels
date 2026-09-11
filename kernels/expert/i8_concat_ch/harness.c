#include "harness_common.h"
#include "kernel_api.h"
#define C1_DIM 3
#define C2_DIM 5
#define H_DIM  8
#define W_DIM  11
#define PLANE  (H_DIM * W_DIM)               /* 88 */
#define A_SIZE (C1_DIM * PLANE)               /* 264 */
#define B_SIZE (C2_DIM * PLANE)               /* 440 */
#define OUT_SIZE ((C1_DIM + C2_DIM) * PLANE)  /* 704 */
static int8_t a[A_SIZE]      HVX_ALIGN;
static int8_t b[B_SIZE]      HVX_ALIGN;
static int8_t out[OUT_SIZE]  HVX_ALIGN;
static int8_t ref[OUT_SIZE]  HVX_ALIGN;
int main(void) {
    uint32_t s = 0x93C4D5u;
    for (int i = 0; i < A_SIZE; i++) a[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < B_SIZE; i++) b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    /* Golden reference */
    for (int c = 0; c < C1_DIM; c++)
        for (int i = 0; i < PLANE; i++) ref[c * PLANE + i] = a[c * PLANE + i];
    for (int c = 0; c < C2_DIM; c++)
        for (int i = 0; i < PLANE; i++) ref[(C1_DIM + c) * PLANE + i] = b[c * PLANE + i];
    for (int i = 0; i < OUT_SIZE; i++) out[i] = (int8_t)0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, C1_DIM, C2_DIM, H_DIM, W_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < OUT_SIZE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    }
    hvx_report(errors, OUT_SIZE, fb, fb >= 0 ? (long)out[fb] : 0, fb >= 0 ? (long)ref[fb] : 0);
    return errors ? 1 : 0;
}
