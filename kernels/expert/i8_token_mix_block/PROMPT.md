# Task: i8_token_mix_block

## Goal
Implement `candidate_kernel` in C using Hexagon HVX intrinsics.
MLP-Mixer token-mixing block: mix tokens along the token axis using two GEMMs.

## Block description
For each channel c independently:
    Step 1 (token-mix GEMM): y1[m] = sat_i8(round_half_away((W[m,:]*X[:,c] + b[m])*mult1, shift1))
    Step 2 (project back):   out[t,c] = sat_i8(round_half_away((W2[t,:]*y1 + b2[t])*mult2, shift2))

This is the token-mixing half of MLP-Mixer: W[D_MIX x TOKENS] mixes across tokens.

## Dims
TOKENS=8, CHANNELS=16, D_MIX=16

## Constraints
- Integer only, no float.
- All params (mult1, shift1, mult2, shift2) are RUNTIME -- do NOT hardcode.
- round_half_away includes the bias before multiplying by mult: biased=(int64)acc+bias; v=biased*mult; ...
- Use HVX_Vector / Q6_V* intrinsics for HVX reward bonus.

## Signature
```c
void candidate_kernel(const int8_t *X, const int8_t *W, const int32_t *b,
                      const int8_t *W2, const int32_t *b2,
                      int8_t *out,
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2);
```
