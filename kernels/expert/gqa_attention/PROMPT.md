Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *Q,
                          const int8_t *K,
                          const int8_t *V,
                          const uint8_t *exp_lut,
                          int8_t *out,
                          int32_t scale_mult, int scale_shift,
                          int32_t av_mult,    int av_shift);

Grouped-Query Attention (GQA) in pure integer arithmetic (NO floating point).
H_Q=4 query heads, H_KV=2 KV heads, GROUP_SIZE=2 (each KV head is shared by 2 query heads).
SEQ=8 tokens, HEAD_DIM=16. All arrays [heads x SEQ x HEAD_DIM] int8, head outermost.

For each query head hq (KV head hkv = hq / GROUP_SIZE):
  Step 1 -- QKT: scores[i,j] = sum_d Q[hq*SEQ*HEAD_DIM + i*HEAD_DIM+d]
                                     * K[hkv*SEQ*HEAD_DIM + j*HEAD_DIM+d]   (int32)
  Step 2 -- Scale to int8 (round-half-away-from-zero):
      v = (int64_t)scores[i,j] * scale_mult
      half = scale_shift>0 ? (1LL<<(scale_shift-1)) : 0
      scaled[i,j] = clamp((v>=0)?(v+half)>>scale_shift:−((−v+half)>>scale_shift), -128, 127)
  Step 3 -- Row-wise integer softmax using runtime exp_lut[256]:
      m = max(scaled[i,:])
      idx[j] = clamp((int16)(scaled[i,j]-m), -255, 0) + 255
      e[j] = exp_lut[idx[j]]
      S = sum(e)
      prob[i,j] = (uint8)((e[j]*255 + S/2) / S)
  Step 4 -- AV: av[i,d] = sum_j prob[i,j] * V[hkv*SEQ*HEAD_DIM + j*HEAD_DIM+d]   (int32)
  Step 5 -- Requant AV: out[hq*SEQ*HEAD_DIM + i*HEAD_DIM+d] =
      clamp(round_half_away(av[i,d]*av_mult, av_shift), -128, 127)

CRITICAL: each query head uses K[hkv] and V[hkv] where hkv = hq / GROUP_SIZE.
  hq=0,1 -> hkv=0;  hq=2,3 -> hkv=1.
exp_lut and all params are runtime inputs -- do NOT hardcode.
Do NOT use floating point. Do NOT write main(). Include <hexagon_types.h> and <hexagon_protos.h>.
Respond with a single complete C code block and CLOSE the fence with ```.
