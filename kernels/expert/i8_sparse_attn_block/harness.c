#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

/* H=1, SEQ=8, HEAD_DIM=16 */

static int8_t  Q_in[H*SEQ*HEAD_DIM]   HVX_ALIGN;
static int8_t  K_in[H*SEQ*HEAD_DIM]   HVX_ALIGN;
static int8_t  V_in[H*SEQ*HEAD_DIM]   HVX_ALIGN;
static uint8_t exp_lut[256]            HVX_ALIGN;
static int8_t  out[H*SEQ*HEAD_DIM]    HVX_ALIGN;
static int8_t  ref[H*SEQ*HEAD_DIM]    HVX_ALIGN;

#define MASK_VAL ((int8_t)(-128))

static int8_t requant_i8(int32_t raw, int32_t mult, int shift) {
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

static void sparse_attn_ref(const int8_t *Q, const int8_t *K, const int8_t *V,
                            const uint8_t *lut, int8_t *r,
                            int window, int32_t smult, int sshift,
                            int32_t amult, int ashift) {
    for (int h = 0; h < H; h++) {
        const int8_t *Qh = Q + h * SEQ * HEAD_DIM;
        const int8_t *Kh = K + h * SEQ * HEAD_DIM;
        const int8_t *Vh = V + h * SEQ * HEAD_DIM;
        int8_t       *rh = r + h * SEQ * HEAD_DIM;

        for (int i = 0; i < SEQ; i++) {
            /* Compute scores with masking */
            int8_t scores[SEQ];
            for (int j = 0; j < SEQ; j++) {
                int dist = i - j; if (dist < 0) dist = -dist;
                if (dist <= window) {
                    int32_t acc = 0;
                    for (int d = 0; d < HEAD_DIM; d++)
                        acc += (int32_t)Qh[i*HEAD_DIM+d] * (int32_t)Kh[j*HEAD_DIM+d];
                    scores[j] = requant_i8(acc, smult, sshift);
                } else {
                    scores[j] = MASK_VAL;   /* masked: minimum int8 */
                }
            }

            /* Softmax */
            int8_t m = scores[0];
            for (int j = 1; j < SEQ; j++) if (scores[j] > m) m = scores[j];

            uint8_t e[SEQ];
            int32_t S = 0;
            for (int j = 0; j < SEQ; j++) {
                int diff = (int)scores[j] - (int)m;
                if (diff < -255) diff = -255;
                e[j] = lut[diff + 255];
                S += (int32_t)e[j];
            }

            uint8_t prob[SEQ];
            int32_t half_S = S / 2;
            for (int j = 0; j < SEQ; j++)
                prob[j] = (uint8_t)(((int32_t)e[j] * 255 + half_S) / S);

            /* AV */
            for (int dv = 0; dv < HEAD_DIM; dv++) {
                int32_t acc = 0;
                for (int j = 0; j < SEQ; j++)
                    acc += (int32_t)prob[j] * (int32_t)Vh[j*HEAD_DIM+dv];
                rh[i*HEAD_DIM+dv] = requant_i8(acc, amult, ashift);
            }
        }
    }
}

/*
 * Sweep: 2 windows x 2 (lut, param) sets = 4 evals.
 * Anti-hardcode: candidate must read runtime window + lut.
 * Plant distinct scores at the window boundary so attending outside matters.
 */
static const int WINDOWS[]     = { 1, 2 };
static const int32_t SMULTS[]  = { 3, 1 };
static const int     SSHIFTS[] = { 7, 4 };
static const int32_t AMULTS[]  = { 5, 2 };
static const int     ASHIFTS[] = { 8, 5 };
#define NWINDOWS 2
#define NPARAMS  2
#define NSETS    (NWINDOWS * NPARAMS)

int main(void) {
    uint32_t s = 0xE1F2A3B4u;

    /* Random Q, K, V */
    for (int i = 0; i < H*SEQ*HEAD_DIM; i++) {
        Q_in[i] = (int8_t)(hvx_lcg(&s) >> 24);
        K_in[i] = (int8_t)(hvx_lcg(&s) >> 24);
        V_in[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Edge cases */
    Q_in[0] = 127;  K_in[0] = 127;   /* large QKT */
    Q_in[1] = -128; K_in[1] = 127;   /* sign edge */
    V_in[0] = 127;  V_in[1] = -128;  /* sign edge in AV */

    /* Plant high-magnitude values at window=1 boundary (i=0, j=2: dist=2 > window=1)
     * so a kernel that incorrectly attends outside the window will get wrong result. */
    for (int d = 0; d < HEAD_DIM; d++) {
        Q_in[0*HEAD_DIM+d] = 64;   /* i=0 */
        K_in[2*HEAD_DIM+d] = 64;   /* j=2: outside window=1 from i=0 */
    }
    V_in[2*HEAD_DIM]   =  127;    /* j=2 V row is extreme */
    V_in[2*HEAD_DIM+1] = -128;

    /* Build 2 distinct exp_lut tables */
    uint8_t luts[NPARAMS][256];
    for (int pi = 0; pi < NPARAMS; pi++) {
        for (int j = 0; j < 256; j++)
            luts[pi][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[pi][255] = 200 + pi * 30;
        luts[pi][0]   = 1;
    }

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int wi = 0; wi < NWINDOWS; wi++) {
        for (int pi = 0; pi < NPARAMS; pi++) {
            int     window = WINDOWS[wi];
            int32_t smult  = SMULTS[pi];
            int     sshift = SSHIFTS[pi];
            int32_t amult  = AMULTS[pi];
            int     ashift = ASHIFTS[pi];

            for (int j = 0; j < 256; j++) exp_lut[j] = luts[pi][j];

            sparse_attn_ref(Q_in, K_in, V_in, exp_lut, ref,
                            window, smult, sshift, amult, ashift);

            for (int i = 0; i < H*SEQ*HEAD_DIM; i++) out[i] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, {
                candidate_kernel(Q_in, K_in, V_in, exp_lut, out,
                window, smult, sshift, amult, ashift);
            });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < H*SEQ*HEAD_DIM; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb = (wi*NPARAMS + pi) * H*SEQ*HEAD_DIM + i;
                        gotv = (long)out[i];
                        expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, NSETS * H*SEQ*HEAD_DIM, fb, gotv, expv);
    return errors ? 1 : 0;
}
