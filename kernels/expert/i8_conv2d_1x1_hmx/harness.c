#include "harness_common.h"
#include "kernel_api.h"

static uint8_t  in[CONV_CIN*CONV_P]     HVX_ALIGN;
static int8_t   W[CONV_COUT*CONV_CIN]   HVX_ALIGN;
static uint16_t out[CONV_COUT*CONV_P]   HVX_ALIGN;
static uint16_t ref[CONV_COUT*CONV_P]   HVX_ALIGN;

int main(void) {
    uint32_t s = 0x71C4u;
    /* in in 0..7 (positive: int8==uint8), W in -3..3. K=C_in=64 -> worst-case
     * |acc| = 64*7*3 = 1344 -> |acc*17/16| = 1428 < 2048, so the 12-bit requant
     * field is exact (no HMX saturation) and bit-exactness holds. */
    for (int i = 0; i < CONV_CIN*CONV_P;   i++) in[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < CONV_COUT*CONV_CIN; i++) W[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int co = 0; co < CONV_COUT; co++)
        for (int p = 0; p < CONV_P; p++) {
            int acc = 0;
            for (int ci = 0; ci < CONV_CIN; ci++)
                acc += (int)in[ci*CONV_P + p] * (int)W[co*CONV_CIN + ci];
            ref[co*CONV_P + p] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }

    for (int i = 0; i < CONV_COUT*CONV_P; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, W, out, CONV_P); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < CONV_COUT*CONV_P; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, CONV_COUT*CONV_P, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
