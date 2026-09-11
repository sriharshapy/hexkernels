#include "harness_common.h"
#include "kernel_api.h"

static uint8_t  in[C_IN*IH*IW]       HVX_ALIGN;
static int8_t   W[C_OUT*C_IN*KH*KW]  HVX_ALIGN;
static int32_t  bias[C_OUT]          HVX_ALIGN;
static int32_t  out[C_OUT*P_OUT]     HVX_ALIGN;
static int32_t  ref[C_OUT*P_OUT]     HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x2F80u;
    /* in in 0..7, W in -3..3. K=72 -> |acc| <= 72*7*3 = 1512 -> |r| <= 1607
     * (12-bit field exact). Bias spans +-2000 so biased crosses zero -> both ReLU
     * branches fire; edge biases force zero / passthrough / zero-bias. */
    for (int i = 0; i < C_IN*IH*IW;       i++) in[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < C_OUT*C_IN*KH*KW; i++) W[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int c = 0; c < C_OUT; c++) bias[c] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -3000; bias[1] = 3000; bias[2] = 0;

    for (int co = 0; co < C_OUT; co++)
        for (int oh = 0; oh < OH; oh++)
            for (int ow = 0; ow < OW; ow++) {
                int acc = 0;
                for (int ci = 0; ci < C_IN; ci++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++)
                            acc += (int)in[ci*IH*IW + (oh*STRIDE + kh)*IW + (ow*STRIDE + kw)]
                                 * (int)W[((co*C_IN + ci)*KH + kh)*KW + kw];
                int r = sx12(hvx_hmx_requant_0x40(acc));
                int biased = r + bias[co];
                ref[co*P_OUT + oh*OW + ow] = biased > 0 ? biased : 0;
            }

    for (int i = 0; i < C_OUT*P_OUT; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, W, bias, out, P_OUT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < C_OUT*P_OUT; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, C_OUT*P_OUT, fb, gotv, expv);
    return errors ? 1 : 0;
}
