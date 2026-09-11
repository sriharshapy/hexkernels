#include "harness_common.h"
#include "kernel_api.h"
#define N 32
#define TILE 1024

static int8_t B[N*N]    HVX_ALIGN;
static int8_t out[TILE] HVX_ALIGN;
static int8_t ref[TILE] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x2C31u;
    /* full signed int8 range -> packing must preserve sign and every value. */
    for (int i = 0; i < N*N; i++) B[i] = (int8_t)(hvx_lcg(&s) & 0xFF);

    /* reference: 4-deep weight packing. Every byte of the 1024-byte tile is set. */
    for (int i = 0; i < TILE; i++) ref[i] = 0;
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            ref[hvx_hmx_i8_wgt_off(k, j)] = B[k*N + j];

    for (int i = 0; i < TILE; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(B, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < TILE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    }
    hvx_report(errors, TILE, fb, gotv, expv);
    return errors ? 1 : 0;
}
