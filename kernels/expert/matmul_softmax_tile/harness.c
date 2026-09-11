#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>
#include <string.h>

static int8_t  Q_mat[SEQ_Q*HEAD_DIM]   HVX_ALIGN;
static int8_t  K_mat[SEQ_K*HEAD_DIM]   HVX_ALIGN;
static uint8_t out[SEQ_Q*SEQ_K]        HVX_ALIGN;
static uint8_t ref[SEQ_Q*SEQ_K]        HVX_ALIGN;
static uint8_t exp_lut[256]            HVX_ALIGN;

static int8_t clamp8(int32_t v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return (int8_t)v;
}

/* Reference: QK^T scaled + rowwise softmax */
static void matmul_softmax_ref(const int8_t *Q, const int8_t *K,
                               uint8_t *r,
                               const uint8_t *lut,
                               int32_t inv_sqrt_d, int shift) {
    /* Step A: scaled QK^T -> int8 clamped scores */
    int8_t scores[SEQ_Q*SEQ_K];
    for (int i = 0; i < SEQ_Q; i++) {
        for (int j = 0; j < SEQ_K; j++) {
            int32_t raw = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                raw += (int32_t)Q[i*HEAD_DIM+d] * (int32_t)K[j*HEAD_DIM+d];
            /* round-half-up fixed-point scale */
            int64_t v    = (int64_t)raw * (int64_t)inv_sqrt_d;
            int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
            int64_t sc;
            if (v >= 0) sc = (v + half) >> shift;
            else        sc = -((-v + half) >> shift);
            scores[i*SEQ_K+j] = clamp8((int32_t)sc);
        }
    }

    /* Step B: rowwise softmax on int8 scores */
    for (int i = 0; i < SEQ_Q; i++) {
        const int8_t *row = scores + i * SEQ_K;
        uint8_t      *or_ = r      + i * SEQ_K;
        /* find row max */
        int8_t m = row[0];
        for (int j = 1; j < SEQ_K; j++) if (row[j] > m) m = row[j];
        /* exp + sum */
        int32_t S = 0;
        for (int j = 0; j < SEQ_K; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            uint8_t e = lut[diff + 255];
            or_[j] = e;
            S += (int32_t)e;
        }
        /* normalize */
        int32_t half_S = S / 2;
        for (int j = 0; j < SEQ_K; j++) {
            int32_t ej = (int32_t)or_[j];
            or_[j] = (uint8_t)((ej * 255 + half_S) / S);
        }
    }
}

/*
 * Anti-cheat sweeps:
 *   - 2 LUT tables
 *   - 2 (inv_sqrt_d, shift) parameter pairs
 * NSETS = 4 total
 */
#define NLUTS  2
#define NPARAMS 2
#define NSETS  (NLUTS * NPARAMS)

static const int32_t INV_SDS[NPARAMS] = { 23, 45 };
static const int     SHIFTS[NPARAMS]  = {  7,  8 };

int main(void) {
    uint32_t s = 0xA7C3E501u;

    uint8_t luts[NLUTS][256];
    for (int k = 0; k < NLUTS; k++) {
        for (int j = 0; j < 256; j++)
            luts[k][j] = (uint8_t)((hvx_lcg(&s) >> 24) & 0xFF);
        luts[k][255] = 220 + (uint8_t)(k * 20);
        luts[k][0]   = 1;
    }

    /* Random Q and K matrices */
    for (int i = 0; i < SEQ_Q*HEAD_DIM; i++) Q_mat[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < SEQ_K*HEAD_DIM; i++) K_mat[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge cases */
    /* Row 0 of Q: all max */
    for (int d = 0; d < HEAD_DIM; d++) Q_mat[0*HEAD_DIM+d] = 127;
    /* Row 1 of Q: all zeros */
    for (int d = 0; d < HEAD_DIM; d++) Q_mat[1*HEAD_DIM+d] = 0;
    /* Row 0 of K: all max (large dot with row 0 Q) */
    for (int d = 0; d < HEAD_DIM; d++) K_mat[0*HEAD_DIM+d] = 127;
    /* Row 1 of K: all min (large negative dot) */
    for (int d = 0; d < HEAD_DIM; d++) K_mat[1*HEAD_DIM+d] = -128;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int kl = 0; kl < NLUTS; kl++) {
        for (int j = 0; j < 256; j++) exp_lut[j] = luts[kl][j];

        for (int kp = 0; kp < NPARAMS; kp++) {
            int32_t inv_sqrt_d = INV_SDS[kp];
            int     shift      = SHIFTS[kp];

            matmul_softmax_ref(Q_mat, K_mat, ref, exp_lut, inv_sqrt_d, shift);

            for (int i = 0; i < SEQ_Q*SEQ_K; i++) out[i] = 0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q_mat, K_mat, out, exp_lut, inv_sqrt_d, shift); });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < SEQ_Q*SEQ_K; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb = (kl * NPARAMS + kp) * SEQ_Q * SEQ_K + i;
                        gotv = (long)out[i];
                        expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, SEQ_Q * SEQ_K * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
