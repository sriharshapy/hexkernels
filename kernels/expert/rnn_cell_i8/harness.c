#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

/* Keep dims small: H*H + H*I matvecs in scalar ref must finish in sim budget.
 * H=32, I=32: 32*(32+32) = 2048 mults per hidden unit, 2048*32 = 65536 total. */
#define H 32
#define I 32

static int8_t  Wx[H*I]      HVX_ALIGN;
static int8_t  Wh[H*H]      HVX_ALIGN;
static int32_t b[H]          HVX_ALIGN;
static int8_t  x_in[I]       HVX_ALIGN;
static int8_t  h_prev[H]     HVX_ALIGN;
static int8_t  h_out[H]      HVX_ALIGN;
static int8_t  ref_out[H]    HVX_ALIGN;
static int8_t  tanh_lut[256] HVX_ALIGN;

/* Requantize: round-half-away-from-zero, saturate int8. */
static int8_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Scalar reference: h_t[j] = tanh_lut[(uint8_t)requant(Wx[j,:]*x + Wh[j,:]*h_prev + b[j])] */
static void rnn_ref(int8_t *ht,
                    const int8_t *Wx_, const int8_t *Wh_,
                    const int32_t *bias,
                    const int8_t *x_, const int8_t *hp,
                    const int8_t *lut,
                    int32_t mult, int shift, int8_t zp) {
    for (int j = 0; j < H; j++) {
        int32_t acc = bias[j];
        for (int k = 0; k < I; k++)
            acc += (int32_t)Wx_[j*I+k] * (int32_t)x_[k];
        for (int k = 0; k < H; k++)
            acc += (int32_t)Wh_[j*H+k] * (int32_t)hp[k];
        int8_t pre = requant(acc, mult, shift, zp);
        ht[j] = lut[(uint8_t)pre];
    }
}

/* Sweep 2 (mult,shift,zp) sets AND 2 LUT tables.
 * A kernel that hardcodes params or the activation table will fail at least one sweep. */
static const int32_t MULTS[]  = {  3,  5 };
static const int     SHIFTS[] = {  7,  5 };
static const int8_t  ZPS[]    = {  0,  2 };
#define NPARAMS 2
#define NLUTS   2

int main(void) {
    uint32_t s = 0xA1B2C3D4u;

    /* Generate weights, bias, input, prev hidden state */
    for (int i = 0; i < H*I; i++) Wx[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H*H; i++) Wh[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H;   i++) b[i]     = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < I;   i++) x_in[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H;   i++) h_prev[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases: extreme weights to exercise requant saturation */
    Wx[0] = 127;  x_in[0] = 127;      /* max positive product */
    Wh[0] = -128; h_prev[0] = 127;    /* max negative product */
    b[0]  = 32767;                     /* large bias */
    b[1]  = -32768;                    /* large negative bias */

    /* Generate 2 independent LUT tables (not real sigmoid/tanh -- arbitrary) */
    int8_t luts[NLUTS][256];
    for (int t = 0; t < NLUTS; t++) {
        for (int k = 0; k < 256; k++)
            luts[t][k] = (int8_t)(hvx_lcg(&s) >> 24);
        /* Ensure index boundaries are hit with distinct values */
        luts[t][0]   = (int8_t)(10 + t * 5);
        luts[t][127] = (int8_t)(-20 - t * 3);
        luts[t][128] = (int8_t)(30 + t * 7);
        luts[t][255] = (int8_t)(-40 - t * 2);
    }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int t = 0; t < NLUTS; t++) {
        /* Copy active LUT */
        for (int k = 0; k < 256; k++) tanh_lut[k] = luts[t][k];

        for (int p = 0; p < NPARAMS; p++) {
            int32_t mult = MULTS[p];
            int     shift = SHIFTS[p];
            int8_t  zp    = ZPS[p];

            /* Build reference */
            rnn_ref(ref_out, Wx, Wh, b, x_in, h_prev, tanh_lut, mult, shift, zp);

            /* Poison output */
            for (int j = 0; j < H; j++) h_out[j] = (int8_t)0xA5;

            /* Run candidate */
            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Wx, Wh, b, x_in, h_prev, h_out, H, I, mult, shift, zp, tanh_lut); });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            /* Compare */
            int sweep_id = t * NPARAMS + p;
            for (int j = 0; j < H; j++) {
                if (h_out[j] != ref_out[j]) {
                    errors++;
                    if (fb < 0) {
                        fb = sweep_id * H + j;
                        gotv = (long)h_out[j];
                        expv = (long)ref_out[j];
                    }
                }
            }
        }
    }

    hvx_report(errors, H * NPARAMS * NLUTS, fb, gotv, expv);
    return errors ? 1 : 0;
}
