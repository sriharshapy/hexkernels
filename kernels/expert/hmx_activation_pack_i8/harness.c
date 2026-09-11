#include "harness_common.h"
#include "kernel_api.h"
#define N 32
#define TILE 2048

static uint8_t A[N*N]    HVX_ALIGN;
static uint8_t out[TILE] HVX_ALIGN;
static uint8_t ref[TILE] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x1A7Fu;
    /* full 0..255 activation range -> pack must preserve every byte value. */
    for (int i = 0; i < N*N; i++) A[i] = (uint8_t)(hvx_lcg(&s) & 0xFF);

    /* reference crouton pack: zero the tile, drop each int8 into its HIGH byte. */
    for (int i = 0; i < TILE; i++) ref[i] = 0;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            ref[hvx_hmx_i8_act_off(i, k)] = A[i*N + k];

    for (int i = 0; i < TILE; i++) *((volatile uint8_t *)&out[i]) = 0xA5; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < TILE; i++) {
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    }
    hvx_report(errors, TILE, fb, gotv, expv);
    return errors ? 1 : 0;
}
