Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const __fp16 *scores, const __fp16 *V,
                          __fp16 *running_max, __fp16 *running_sum, __fp16 *acc,
                          int BLK, int D);

Flash-attention-style ONLINE SOFTMAX update: fold ONE new block of BLK
key/value positions into an existing partial online-softmax state
(`*running_max`, `*running_sum`, `acc[0..D)`) for a single query row,
updating that state IN PLACE.

scores: [BLK] fp16 raw attention scores for the new block.
V: [BLK x D] fp16, row-major -- the new block's value vectors.

Exact recurrence (do the internal math in `float`; only the
persisted `running_max`/`running_sum`/`acc[]` boundary values are fp16):

```
m_old = (float)*running_max;  l_old = (float)*running_sum;
m_blk = max_j scores[j]                              for j in [0,BLK)
m_new = max(m_old, m_blk)
corr  = expf(m_old - m_new)          // rescale factor for the OLD accumulator (1.0 if m_new==m_old)
p[j]  = expf((float)scores[j] - m_new)    for j in [0,BLK)
l_new = l_old * corr + sum_j p[j]
for d in [0,D): acc[d] = (hvx_hf)( (float)acc[d]*corr + sum_j p[j] * (float)V[j*D+d] )
*running_max = (hvx_hf)m_new;  *running_sum = (hvx_hf)l_new;
```

BLK=24, D=48 (neither a multiple of 64 fp16-lanes-per-HVX-vector -- real
tail path on both the score-block axis and the D axis). HVX has NO
vectorized exp -- the exp calls are necessarily scalar, but the block max
-reduce, the accumulator rescale-by-corr, and the weighted-value
accumulation over D ARE vectorizable.

Output is compared to an independent float32 scalar reference with an
fp16 tolerance (HVX float arithmetic is non-IEEE qfloat, not bit-exact).
This is THE classic online-softmax update -- get the rescale-by-corr of
the OLD accumulator right, it is the step every naive implementation
forgets.

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
