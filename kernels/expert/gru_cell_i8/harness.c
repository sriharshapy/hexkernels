#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

/* Small dims: H=24, I=24, C=48. Each gate does 24*48=1152 mults.
 * 3 gates * 1152 * NPARAMS * NLUTS sweeps well within 60s sim budget. */
#define H 24
#define I 24
#define C (I+H)

static int8_t  Wz[H*C]      HVX_ALIGN;
static int8_t  Wr[H*C]      HVX_ALIGN;
static int8_t  Wn[H*C]      HVX_ALIGN;
static int32_t bz[H]         HVX_ALIGN;
static int32_t br[H]         HVX_ALIGN;
static int32_t bn[H]         HVX_ALIGN;
static int8_t  x_in[I]       HVX_ALIGN;
static int8_t  h_prev[H]     HVX_ALIGN;
static int8_t  h_out[H]      HVX_ALIGN;
static int8_t  ref_out[H]    HVX_ALIGN;
static int8_t  sig_lut[256]  HVX_ALIGN;
static int8_t  tanh_lut[256] HVX_ALIGN;

static int8_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

static int8_t clamp8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* Scalar reference for one GRU step. */
static void gru_ref(int8_t *ht,
                    const int8_t *Wz_, const int8_t *Wr_, const int8_t *Wn_,
                    const int32_t *bz_, const int32_t *br_, const int32_t *bn_,
                    const int8_t *x_, const int8_t *hp,
                    const int8_t *slut, const int8_t *tlut,
                    int32_t mult, int shift, int8_t zp) {
    int8_t z[H], r[H], rh[H], n[H];

    /* Update gate z */
    for (int j = 0; j < H; j++) {
        int32_t acc = bz_[j];
        for (int k = 0; k < I; k++)  acc += (int32_t)Wz_[j*C+k]   * (int32_t)x_[k];
        for (int k = 0; k < H; k++)  acc += (int32_t)Wz_[j*C+I+k] * (int32_t)hp[k];
        z[j] = slut[(uint8_t)requant(acc, mult, shift, zp)];
    }

    /* Reset gate r */
    for (int j = 0; j < H; j++) {
        int32_t acc = br_[j];
        for (int k = 0; k < I; k++)  acc += (int32_t)Wr_[j*C+k]   * (int32_t)x_[k];
        for (int k = 0; k < H; k++)  acc += (int32_t)Wr_[j*C+I+k] * (int32_t)hp[k];
        r[j] = slut[(uint8_t)requant(acc, mult, shift, zp)];
    }

    /* Gated recurrent: rh[j] = clamp(((r[j]+128)*h_prev[j] + 64) >> 7) */
    for (int j = 0; j < H; j++) {
        int32_t blend = ((int32_t)(r[j] + 128) * (int32_t)hp[j] + 64) >> 7;
        rh[j] = clamp8(blend);
    }

    /* Candidate hidden n (recurrent uses rh, not h_prev) */
    for (int j = 0; j < H; j++) {
        int32_t acc = bn_[j];
        for (int k = 0; k < I; k++)  acc += (int32_t)Wn_[j*C+k]   * (int32_t)x_[k];
        for (int k = 0; k < H; k++)  acc += (int32_t)Wn_[j*C+I+k] * (int32_t)rh[k];
        n[j] = tlut[(uint8_t)requant(acc, mult, shift, zp)];
    }

    /* Output blend: h_t[j] = clamp(((128-z[j])*h_prev[j] + (z[j]+128)*n[j] + 128) >> 8) */
    for (int j = 0; j < H; j++) {
        int32_t blend = ((int32_t)(128 - z[j]) * (int32_t)hp[j]
                       + (int32_t)(z[j] + 128) * (int32_t)n[j] + 128) >> 8;
        ht[j] = clamp8(blend);
    }
}

/* Sweep 2 (mult,shift,zp) sets AND 2 (sig_lut,tanh_lut) pairs.
 * Anti-cheat: hardcoding requant params OR gate order OR LUT fails a sweep. */
static const int32_t MULTS[]  = {  3,  7 };
static const int     SHIFTS[] = {  7,  6 };
static const int8_t  ZPS[]    = {  0,  1 };
#define NPARAMS 2
#define NLUTS   2

int main(void) {
    uint32_t s = 0xB5C6D7E8u;

    for (int i = 0; i < H*C; i++) Wz[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H*C; i++) Wr[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H*C; i++) Wn[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H;   i++) bz[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < H;   i++) br[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < H;   i++) bn[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < I;   i++) x_in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H;   i++) h_prev[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases: extreme inputs to exercise gate saturation and blend boundaries */
    Wz[0] = 127;  x_in[0] = 127;     /* max product in update gate */
    Wr[0] = -128; h_prev[0] = 127;   /* max neg product in reset gate */
    bz[0] = 32767;  bz[1] = -32768;  /* large biases */
    h_prev[1] = 0;                    /* zero prev state edge */
    h_prev[2] = -128;                 /* min prev state */

    /* Generate 2 independent LUT pairs */
    int8_t sluts[NLUTS][256];
    int8_t tluts[NLUTS][256];
    for (int t = 0; t < NLUTS; t++) {
        for (int k = 0; k < 256; k++) {
            sluts[t][k] = (int8_t)(hvx_lcg(&s) >> 24);
            tluts[t][k] = (int8_t)(hvx_lcg(&s) >> 24);
        }
        /* Boundary pins to ensure those indices hit distinct values */
        sluts[t][0]   = (int8_t)(-60 + t * 11);
        sluts[t][128] = (int8_t)( 50 + t * 13);
        sluts[t][255] = (int8_t)( 30 + t * 7);
        tluts[t][0]   = (int8_t)( 10 + t * 5);
        tluts[t][128] = (int8_t)(-40 - t * 9);
        tluts[t][255] = (int8_t)( 20 + t * 3);
    }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int t = 0; t < NLUTS; t++) {
        for (int k = 0; k < 256; k++) {
            sig_lut[k]  = sluts[t][k];
            tanh_lut[k] = tluts[t][k];
        }
        for (int p = 0; p < NPARAMS; p++) {
            int32_t mult  = MULTS[p];
            int     shift = SHIFTS[p];
            int8_t  zp    = ZPS[p];

            gru_ref(ref_out, Wz, Wr, Wn, bz, br, bn, x_in, h_prev,
                    sig_lut, tanh_lut, mult, shift, zp);

            for (int j = 0; j < H; j++) h_out[j] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, {
                candidate_kernel(Wz, Wr, Wn, bz, br, bn, x_in, h_prev, h_out,
                H, I, mult, shift, zp, sig_lut, tanh_lut);
            });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

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
