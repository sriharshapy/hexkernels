# Task: i8_sparse_attn_block

## Goal
Implement `candidate_kernel` in C using Hexagon HVX intrinsics.
Compute sparse (banded) attention: each query row i attends only within a window of `window` positions.

## Block description
For each query row i and key column j:
    if |i - j| <= window: score[j] = sat_i8(round_half_away(QKT[i,j]*smult, sshift))
    else:                 score[j] = -128  (MASK_VAL -- minimum int8)
Then softmax over all SEQ positions (including masked ones which get exp_lut[low_idx]).
Output: sat_i8(round_half_away(sum_j(prob[j]*V[j,d]) * amult, ashift)).

## Dims
H=1, SEQ=8, HEAD_DIM=16

## Constraints
- Integer only, no float.
- window is a RUNTIME param -- do NOT hardcode (swept: 1, 2).
- exp_lut[256] and all quant params are RUNTIME -- do NOT hardcode.
- Masked positions use MASK_VAL=-128, NOT excluded from the softmax sum.
- Use HVX_Vector / Q6_V* intrinsics for HVX reward bonus.

## Signature
```c
void candidate_kernel(const int8_t *Q, const int8_t *K, const int8_t *V,
                      const uint8_t *exp_lut, int8_t *out,
                      int window,
                      int32_t smult, int sshift,
                      int32_t amult, int ashift);
```
