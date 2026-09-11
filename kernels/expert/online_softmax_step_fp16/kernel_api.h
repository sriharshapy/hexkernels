#ifndef KERNEL_API_H
#define KERNEL_API_H
typedef __fp16 hvx_hf;
/*
 * Flash-attention-style ONLINE SOFTMAX update step (fp16).
 *
 * Processes ONE new block of BLK key/value positions against a query row
 * that already carries partial online-softmax state (*running_max,
 * *running_sum, acc[0..D)), updating that state IN PLACE.
 *
 * scores: [BLK] fp16 -- raw attention scores for the new block.
 * V:      [BLK x D] fp16, row-major -- the new block's value vectors.
 * running_max, running_sum: scalar fp16 state (in/out).
 * acc: [D] fp16 -- running weighted-value accumulator (in/out).
 *
 * Exact recurrence (internal math in float; only the boundary reads/writes
 * of running_max/running_sum/acc are fp16):
 *   m_old = (float)*running_max;  l_old = (float)*running_sum;
 *   m_blk = max_j scores[j]                              for j in [0,BLK)
 *   m_new = max(m_old, m_blk)
 *   corr  = expf(m_old - m_new)                          (rescale factor for the OLD accumulator; 1.0 if m_new==m_old)
 *   p[j]  = expf((float)scores[j] - m_new)                for j in [0,BLK)
 *   l_new = l_old * corr + sum_j p[j]
 *   for d in [0,D): acc[d] = (hvx_hf)( (float)acc[d]*corr + sum_j p[j] * (float)V[j*D+d] )
 *   *running_max = (hvx_hf)m_new;  *running_sum = (hvx_hf)l_new;
 *
 * BLK=24, D=48 (neither a multiple of 64 fp16-lanes-per-HVX-vector -- tail
 * path on both axes). Output compared with fp16 tolerance (HVX float
 * arithmetic is non-IEEE qfloat).
 */
void candidate_kernel(const hvx_hf *scores, const hvx_hf *V,
                      hvx_hf *running_max, hvx_hf *running_sum, hvx_hf *acc,
                      int BLK, int D);
#endif /* KERNEL_API_H */
