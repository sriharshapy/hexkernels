#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Sparse (banded) attention: each query row i attends ONLY to key columns
 * j where |i - j| <= window (causal-style: j in [max(0,i-window), min(SEQ-1,i+window)]).
 * Positions outside the window get a MASK value (-128) before softmax, effectively
 * zeroing their exp contribution.
 *
 * Algorithm for each row i:
 *   1. For j in [0, SEQ):
 *        if |i-j| <= window:
 *            raw = sum_d Q[i,d]*K[j,d]  (int32)
 *            score[j] = sat_i8(round_half_away(raw * smult, sshift))
 *        else:
 *            score[j] = MASK_VAL   (= -128, i.e. forced minimum)
 *   2. Softmax via exp_lut:
 *        m = max(score[0..SEQ-1])
 *        e[j] = exp_lut[clamp(score[j]-m, -255, 0) + 255]
 *        S = sum(e)
 *        prob[j] = (e[j]*255 + S/2) / S  (uint8)
 *   3. out[i,d] = sat_i8(round_half_away( sum_j(prob[j]*V[j,d]) * amult, ashift ))
 *
 * NOTE on masked positions: MASK_VAL=-128 => idx=clamp(-128-m,-255,0)+255.
 * When m >= -128 (true unless all scores are -128), masked positions contribute
 * exp_lut[0..1] -- NOT zero. The reference DOES include masked positions in S and
 * the final weighted sum. This is intentional: the reference is a pinned formula.
 *
 * window is a runtime param (DO NOT hardcode). Swept: {1, 2}.
 * H=1 (single head), SEQ=8, HEAD_DIM=16.
 */
#define H        1
#define SEQ      8
#define HEAD_DIM 16

void candidate_kernel(const int8_t  *Q,        /* [H x SEQ x HEAD_DIM] int8 */
                      const int8_t  *K,        /* [H x SEQ x HEAD_DIM] int8 */
                      const int8_t  *V,        /* [H x SEQ x HEAD_DIM] int8 */
                      const uint8_t *exp_lut,  /* [256] uint8 */
                      int8_t        *out,      /* [H x SEQ x HEAD_DIM] int8 */
                      int window,
                      int32_t smult, int sshift,
                      int32_t amult, int ashift);
#endif /* KERNEL_API_H */
