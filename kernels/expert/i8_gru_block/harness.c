#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

/* T=3 timesteps, H=16, I=16, C=32. Fused Wzr [2H x C], separate Wn [H x C]. */
#define T 3
#define H 16
#define I 16
#define C (I+H)

static int8_t  Wzr[2*H*C]   HVX_ALIGN;
static int8_t  Wn [H*C]     HVX_ALIGN;
static int32_t bzr[2*H]     HVX_ALIGN;
static int32_t bn [H]       HVX_ALIGN;
static int8_t  x_seq[T*I]   HVX_ALIGN;
static int8_t  h0[H]        HVX_ALIGN;
static int8_t  h_out[T*H]   HVX_ALIGN;
static int8_t  ref_out[T*H] HVX_ALIGN;
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

/* Scalar reference for the full T-step GRU block. */
static void gru_block_ref(int8_t *hout,
                    const int8_t *Wzr_, const int8_t *Wn_,
                    const int32_t *bzr_, const int32_t *bn_,
                    const int8_t *xseq, const int8_t *h0_,
                    const int8_t *slut, const int8_t *tlut,
                    int32_t mult, int shift, int8_t zp) {
    int8_t hprev[H];
    for (int j = 0; j < H; j++) hprev[j] = h0_[j];

    for (int t = 0; t < T; t++) {
        const int8_t *xt = xseq + t*I;
        int8_t z[H], r[H], rh[H], n[H];

        /* Fused update/reset gates over concat_xh = [xt | hprev]. */
        for (int j = 0; j < 2*H; j++) {
            int32_t acc = bzr_[j];
            for (int k = 0; k < I; k++) acc += (int32_t)Wzr_[j*C+k]   * (int32_t)xt[k];
            for (int k = 0; k < H; k++) acc += (int32_t)Wzr_[j*C+I+k] * (int32_t)hprev[k];
            int8_t q = requant(acc, mult, shift, zp);
            if (j < H) z[j]   = slut[(uint8_t)q];
            else       r[j-H] = slut[(uint8_t)q];
        }

        for (int j = 0; j < H; j++) {
            int32_t blend = ((int32_t)(r[j] + 128) * (int32_t)hprev[j] + 64) >> 7;
            rh[j] = clamp8(blend);
        }

        /* Candidate hidden n over concat_xrh = [xt | rh]. */
        for (int j = 0; j < H; j++) {
            int32_t acc = bn_[j];
            for (int k = 0; k < I; k++) acc += (int32_t)Wn_[j*C+k]   * (int32_t)xt[k];
            for (int k = 0; k < H; k++) acc += (int32_t)Wn_[j*C+I+k] * (int32_t)rh[k];
            n[j] = tlut[(uint8_t)requant(acc, mult, shift, zp)];
        }

        for (int j = 0; j < H; j++) {
            int32_t blend = ((int32_t)(128 - z[j]) * (int32_t)hprev[j]
                           + (int32_t)(z[j] + 128) * (int32_t)n[j] + 128) >> 8;
            int8_t ht = clamp8(blend);
            hout[t*H+j] = ht;
            hprev[j] = ht;
        }
    }
}

/* Sweep 2 (mult,shift,zp) sets AND 2 (sig_lut,tanh_lut) pairs -- same
 * anti-hardcode discipline as gru_cell_i8. */
static const int32_t MULTS[]  = {  3,  7 };
static const int     SHIFTS[] = {  7,  6 };
static const int8_t  ZPS[]    = {  0,  1 };
#define NPARAMS 2
#define NLUTS   2

int main(void) {
    uint32_t s = 0xC0FFEE31u;

    for (int i = 0; i < 2*H*C; i++) Wzr[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H*C;   i++) Wn[i]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < 2*H;   i++) bzr[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < H;     i++) bn[i]  = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < T*I;   i++) x_seq[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < H;     i++) h0[i]    = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases: extreme inputs to exercise gate saturation and blend boundaries. */
    Wzr[0] = 127;  x_seq[0] = 127;        /* max product in update gate, step 0 */
    Wzr[H*C] = -128; h0[0] = 127;         /* max neg product in reset gate row 0 (j=H) */
    bzr[0] = 32767;  bzr[1] = -32768;     /* large biases */
    h0[1] = 0;                             /* zero prev state edge */
    h0[2] = -128;                          /* min prev state */
    x_seq[T*I - 1] = -128;                 /* extreme input on the LAST timestep */

    int8_t sluts[NLUTS][256];
    int8_t tluts[NLUTS][256];
    for (int t = 0; t < NLUTS; t++) {
        for (int k = 0; k < 256; k++) {
            sluts[t][k] = (int8_t)(hvx_lcg(&s) >> 24);
            tluts[t][k] = (int8_t)(hvx_lcg(&s) >> 24);
        }
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

            gru_block_ref(ref_out, Wzr, Wn, bzr, bn, x_seq, h0,
                          sig_lut, tanh_lut, mult, shift, zp);

            for (int j = 0; j < T*H; j++) h_out[j] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, {
                candidate_kernel(Wzr, Wn, bzr, bn, x_seq, h0, h_out,
                T, H, I, mult, shift, zp, sig_lut, tanh_lut);
            });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            int sweep_id = t * NPARAMS + p;
            for (int j = 0; j < T*H; j++) {
                if (h_out[j] != ref_out[j]) {
                    errors++;
                    if (fb < 0) {
                        fb = sweep_id * T * H + j;
                        gotv = (long)h_out[j];
                        expv = (long)ref_out[j];
                    }
                }
            }
        }
    }

    hvx_report(errors, T * H * NPARAMS * NLUTS, fb, gotv, expv);
    return errors ? 1 : 0;
}
