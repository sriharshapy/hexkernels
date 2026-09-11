#include "harness_common.h"
#include "kernel_api.h"

static uint8_t  in[C_IN*IH*IW]          HVX_ALIGN;
static int8_t   W[C_OUT*C_IN*KH*KW]     HVX_ALIGN;
static uint16_t out[C_OUT*P_OUT]        HVX_ALIGN;
static uint16_t ref[C_OUT*P_OUT]        HVX_ALIGN;

int main(void) {
    uint32_t s = 0x5CE1u;
    /* in in 0..7 (positive: int8==uint8), W in -3..3. K=72 -> worst-case
     * |acc| = 72*7*3 = 1512 -> |acc*17/16| = 1607 < 2048, so the 12-bit requant
     * field is exact (no HMX saturation) and bit-exactness holds. */
    for (int i = 0; i < C_IN*IH*IW;      i++) in[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < C_OUT*C_IN*KH*KW; i++) W[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    for (int co = 0; co < C_OUT; co++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int acc = 0;
                for (int ci = 0; ci < C_IN; ci++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            acc += (int)in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)]
                                 * (int)W[((co*C_IN + ci)*KH + kh)*KW + kw];
                ref[co*P_OUT + oh*OW + ow] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }

    for (int i = 0; i < C_OUT*P_OUT; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, W, out, P_OUT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < C_OUT*P_OUT; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, C_OUT*P_OUT, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
