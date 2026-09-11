/*
 * GQA Attention HVX kernel.
 * H_Q=4, H_KV=2, GROUP_SIZE=2, SEQ=8, HEAD_DIM=16.
 *
 * Strategy:
 *   QKT: HEAD_DIM=16 = 4 k-groups of 4. Pretranspose K head per-use.
 *        Kt block for k-group kg: 128 bytes, word lane j holds K[j][kg*4..+3].
 *        SEQ=8 so only 8 word-lanes used; rest zero.
 *        vrmpyacc with splat of Q[i][kg*4..+3] -> 8 int32 accumulators.
 *   Scale: scalar requant (8 values, tiny cost).
 *   Softmax: scalar (8 values per row, negligible).
 *   AV: SEQ=8 = 2 k-groups of 4. HEAD_DIM=16 fits in one HVX_Vector (16 words).
 *       Pretranspose V: Vt[kg][d*4+q] = V[(kg*4+q)*HEAD_DIM + d], covers 16 d-values.
 *       Each kg needs one 128-byte block (16 d * 4 bytes = 64 bytes, padded to 128).
 *       vrmpyacc with splat of prob[i][kg*4..+3] -> 16 int32 accumulators -> requant.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

#define H_Q        4
#define H_KV       2
#define GROUP_SIZE 2
#define SEQ        8
#define HEAD_DIM   16

/* HEAD_DIM=16: 4 groups of 4 */
#define QKT_KG    4   /* HEAD_DIM/4 */
/* SEQ=8: 2 groups of 4 */
#define AV_KG     2   /* SEQ/4 */

void candidate_kernel(const int8_t *Q,
                      const int8_t *K,
                      const int8_t *V,
                      const uint8_t *exp_lut,
                      int8_t *out,
                      int32_t scale_mult, int scale_shift,
                      int32_t av_mult,    int av_shift)
{
    /*
     * Kt: transposed K for one KV head.
     * QKT_KG blocks x 128 bytes each.
     * Block kg: 32 word-lanes; lane j = K[j][kg*4..+3] (j=0..7 used, rest 0).
     */
    static int8_t __attribute__((aligned(128))) Kt[QKT_KG * 128];

    /*
     * Vt: transposed V for one KV head.
     * AV_KG blocks x 128 bytes each.
     * Block kg: lane d (0..15) = { V[(kg*4+0)*HEAD_DIM+d],
     *                              V[(kg*4+1)*HEAD_DIM+d],
     *                              V[(kg*4+2)*HEAD_DIM+d],
     *                              V[(kg*4+3)*HEAD_DIM+d] }
     * 16 d-values x 4 bytes = 64 bytes; upper 64 bytes zero.
     */
    static int8_t __attribute__((aligned(128))) Vt[AV_KG * 128];

    /* 32 int32 scratch for QKT accumulator extraction */
    int32_t __attribute__((aligned(128))) accbuf[32];

    /* requant shift constants */
    int64_t sc_half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0LL;
    int64_t av_half = (av_shift   > 0) ? ((int64_t)1 << (av_shift   - 1)) : 0LL;

    /* Softmax scores (int8) and probabilities (uint8) for one query row */
    int8_t  scores[SEQ];  /* one i-row of QKT scores */
    uint8_t probs[SEQ];   /* one i-row of softmax output */

    /* AV int32 accumulator (HEAD_DIM=16 words) */
    int32_t __attribute__((aligned(128))) av_acc[32]; /* 32 word-lanes, 16 used */

    for (int hq = 0; hq < H_Q; hq++) {
        int hkv = hq / GROUP_SIZE;
        const int8_t *Qh  = Q + hq  * SEQ * HEAD_DIM;
        const int8_t *Kh  = K + hkv * SEQ * HEAD_DIM;
        const int8_t *Vh  = V + hkv * SEQ * HEAD_DIM;
        int8_t       *outh = out + hq * SEQ * HEAD_DIM;

        /* --- Build Kt for this KV head --- */
        for (int kg = 0; kg < QKT_KG; kg++) {
            int8_t *blk = Kt + kg * 128;
            memset(blk, 0, 128);
            int k0 = kg * 4;
            for (int j = 0; j < SEQ; j++) {
                blk[j*4+0] = Kh[j*HEAD_DIM + k0+0];
                blk[j*4+1] = Kh[j*HEAD_DIM + k0+1];
                blk[j*4+2] = Kh[j*HEAD_DIM + k0+2];
                blk[j*4+3] = Kh[j*HEAD_DIM + k0+3];
            }
        }

        /* --- Build Vt for this KV head --- */
        for (int kg = 0; kg < AV_KG; kg++) {
            int8_t *blk = Vt + kg * 128;
            memset(blk, 0, 128);
            /* d=0..15, q=0..3 */
            for (int d = 0; d < HEAD_DIM; d++) {
                blk[d*4+0] = Vh[(kg*4+0)*HEAD_DIM + d];
                blk[d*4+1] = Vh[(kg*4+1)*HEAD_DIM + d];
                blk[d*4+2] = Vh[(kg*4+2)*HEAD_DIM + d];
                blk[d*4+3] = Vh[(kg*4+3)*HEAD_DIM + d];
            }
        }

        /* --- Per query row i --- */
        for (int i = 0; i < SEQ; i++) {
            const int8_t *Qrow = Qh + i * HEAD_DIM;

            /* == Step 1: QKT via vrmpy == */
            HVX_Vector qkt_acc = Q6_V_vzero();
            for (int kg = 0; kg < QKT_KG; kg++) {
                int k0 = kg * 4;
                /* Pack Q bytes as uint8 into int32 word (vrmpy treats bytes signed) */
                int32_t q4 = ((uint32_t)(uint8_t)Qrow[k0+0])
                           | ((uint32_t)(uint8_t)Qrow[k0+1] <<  8)
                           | ((uint32_t)(uint8_t)Qrow[k0+2] << 16)
                           | ((uint32_t)(uint8_t)Qrow[k0+3] << 24);
                HVX_Vector vq = Q6_V_vsplat_R(q4);
                HVX_Vector vk = *(const HVX_Vector *)(Kt + kg * 128);
                qkt_acc = Q6_Vw_vrmpyacc_VwVbVb(qkt_acc, vq, vk);
            }
            *(HVX_Vector *)accbuf = qkt_acc;

            /* == Step 2: Scalar requant -> int8 scores for this row == */
            for (int j = 0; j < SEQ; j++) {
                int64_t r = (int64_t)accbuf[j] * (int64_t)scale_mult;
                int64_t q;
                if (r >= 0) q = (r + sc_half) >> scale_shift;
                else        q = -((-r + sc_half) >> scale_shift);
                if (q >  127) q =  127;
                if (q < -128) q = -128;
                scores[j] = (int8_t)q;
            }

            /* == Step 3: Row-wise softmax (scalar, SEQ=8 is tiny) == */
            int8_t m = scores[0];
            for (int j = 1; j < SEQ; j++) if (scores[j] > m) m = scores[j];
            int32_t S = 0;
            for (int j = 0; j < SEQ; j++) {
                int diff = (int)scores[j] - (int)m;
                if (diff < -255) diff = -255;
                probs[j] = exp_lut[diff + 255];
                S += (int32_t)probs[j];
            }
            int32_t half_S = S / 2;
            for (int j = 0; j < SEQ; j++) {
                int32_t ej = (int32_t)probs[j];
                probs[j] = (uint8_t)((ej * 255 + half_S) / S);
            }

            /* == Step 4: AV via vrmpy ==
             * SEQ=8 = 2 k-groups of 4. HEAD_DIM=16 -> one 128B vector (16 words).
             * Vt block kg: bytes [d*4+q] = V[(kg*4+q)*HEAD_DIM+d].
             * probs are uint8; use Q6_Vw_vrmpyacc_VwVubVb (unsigned x signed).
             * Splat probs[kg*4..+3] as int32 word into Vu (unsigned), Vt is Vv (signed V).
             */
            HVX_Vector av_vec = Q6_V_vzero();
            for (int kg = 0; kg < AV_KG; kg++) {
                int k0 = kg * 4;
                /* Pack uint8 prob bytes into word for splat */
                int32_t p4 = ((uint32_t)probs[k0+0])
                           | ((uint32_t)probs[k0+1] <<  8)
                           | ((uint32_t)probs[k0+2] << 16)
                           | ((uint32_t)probs[k0+3] << 24);
                HVX_Vector vp = Q6_V_vsplat_R(p4);  /* Vu: unsigned probs */
                HVX_Vector vv = *(const HVX_Vector *)(Vt + kg * 128); /* Vv: signed V */
                av_vec = Q6_Vw_vrmpyacc_VwVubVb(av_vec, vp, vv);
            }
            *(HVX_Vector *)av_acc = av_vec;

            /* == Step 5: Requant AV -> int8 output == */
            int8_t *orow = outh + i * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                int64_t r = (int64_t)av_acc[d] * (int64_t)av_mult;
                int64_t q;
                if (r >= 0) q = (r + av_half) >> av_shift;
                else        q = -((-r + av_half) >> av_shift);
                if (q >  127) q =  127;
                if (q < -128) q = -128;
                orow[d] = (int8_t)q;
            }
        }
    }
}
