#include "harness_common.h"
#include "kernel_api.h"

static uint8_t in[C_IN*L_IN]        HVX_ALIGN;
static int8_t  W[C_OUT*C_IN*KW]     HVX_ALIGN;
static int32_t bias[C_OUT]          HVX_ALIGN;
static int8_t  out[C_OUT*OL]        HVX_ALIGN;
static int8_t  ref[C_OUT*OL]        HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

int main(void) {
    uint32_t s = 0x9A17u;
    /* in in 0..3, W in -1..1, K=80 -> |acc| <= 80*3 = 240 -> |r| <= 255 (12-bit field
     * exact). bias in +-40 with edges so the int8 output spans the range and both
     * saturation clamps fire. */
    for (int i = 0; i < C_IN*L_IN;      i++) in[i] = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < C_OUT*C_IN*KW;  i++) W[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1);
    for (int c = 0; c < C_OUT; c++) bias[c] = (int32_t)((int)(hvx_lcg(&s) % 81) - 40);
    bias[0] = -200; bias[1] = 200; bias[2] = 0;   /* force -128 / +127 / no-bias */

    for (int co = 0; co < C_OUT; co++)
        for (int ol = 0; ol < OL; ol++) {
            int acc = 0;
            for (int ci = 0; ci < C_IN; ci++)
                for (int kw = 0; kw < KW; kw++)
                    acc += (int)in[ci*L_IN + ol + kw] * (int)W[(co*C_IN + ci)*KW + kw];
            int v = sx12(hvx_hmx_requant_0x40(acc)) + bias[co];
            if (v > 127) v = 127; if (v < -128) v = -128;
            ref[co*OL + ol] = (int8_t)v;
        }

    for (int i = 0; i < C_OUT*OL; i++) *((volatile int8_t *)&out[i]) = (int8_t)0xA5;

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, W, bias, out, OL); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < C_OUT*OL; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, C_OUT*OL, fb, gotv, expv);
    return errors ? 1 : 0;
}
