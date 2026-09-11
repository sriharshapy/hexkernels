#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Grouped-Query Attention (GQA): H_Q query heads share H_KV key/value heads.
 * Each KV head is shared by (H_Q / H_KV) query heads (groups must divide evenly).
 *
 * Layout (all row-major, head is outermost):
 *   Q:   [H_Q  x SEQ x HEAD_DIM] int8
 *   K:   [H_KV x SEQ x HEAD_DIM] int8
 *   V:   [H_KV x SEQ x HEAD_DIM] int8
 *   out: [H_Q  x SEQ x HEAD_DIM] int8
 *
 * Per query head h_q (KV head = h_q / GROUP_SIZE):
 *   1. QKT:   scores[i,j] = sum_d Q[h_q*SEQ*HEAD_DIM + i*HEAD_DIM+d]
 *                                * K[h_kv*SEQ*HEAD_DIM + j*HEAD_DIM+d]   (int32)
 *   2. Scale: scaled[i,j] = clamp(round_half_away((int64_t)scores[i,j]*scale_mult, scale_shift), -128, 127)
 *   3. Softmax (integer, row-wise) using exp_lut[256]:
 *              m      = max(scaled[i,:])
 *              idx[j] = clamp((int16)(scaled[i,j]-m), -255, 0) + 255
 *              e[j]   = exp_lut[idx[j]]
 *              S      = sum(e)
 *              prob[i,j] = (uint8)((e[j]*255 + S/2) / S)
 *   4. AV:    av[i,d] = sum_j prob[i,j] * V[h_kv*SEQ*HEAD_DIM + j*HEAD_DIM+d]  (int32)
 *   5. Requant: out[h_q*SEQ*HEAD_DIM + i*HEAD_DIM+d] =
 *              clamp(round_half_away((int64_t)av[i,d]*av_mult, av_shift), -128, 127)
 *
 * Runtime params (all swept, do NOT hardcode):
 *   scale_mult, scale_shift, av_mult, av_shift
 *   exp_lut: 256 uint8 entries
 *
 * H_Q=4, H_KV=2, GROUP_SIZE=2, SEQ=8, HEAD_DIM=16.
 */
#define H_Q        4
#define H_KV       2
#define GROUP_SIZE 2   /* H_Q / H_KV */
#define SEQ        8
#define HEAD_DIM   16

void candidate_kernel(const int8_t *Q,
                      const int8_t *K,
                      const int8_t *V,
                      const uint8_t *exp_lut,
                      int8_t *out,
                      int32_t scale_mult, int scale_shift,
                      int32_t av_mult,    int av_shift);
#endif /* KERNEL_API_H */
