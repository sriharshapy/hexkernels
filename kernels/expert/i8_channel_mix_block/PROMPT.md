# Task: i8_channel_mix_block

## Goal
Implement `candidate_kernel` in C using Hexagon HVX intrinsics.
MLP-Mixer channel-mixing block: expand channels, apply GELU via LUT, project back.

## Block description
For each token t:
    Step 1 (expand): sat1[m] = sat_i8(round_half_away((W1[m,:]*X[t,:]+b1[m])*mult1, shift1))
                     y1[m]   = gelu_lut[(uint8)(sat1[m] + 128)]
    Step 2 (project): out[t,c] = sat_i8(round_half_away((W2[c,:]*y1+b2[c])*mult2, shift2))

This is the channel-mixing half of MLP-Mixer with integer GELU approximation via a runtime LUT.

## Dims
TOKENS=8, CHANNELS=16, D_FF=16

## Constraints
- Integer only, no float.
- gelu_lut[256] is a RUNTIME int8 table -- do NOT hardcode.
- All params (mult1, shift1, mult2, shift2) are RUNTIME -- do NOT hardcode.
- round_half_away: biased=(int64)acc+bias; v=biased*mult; half=(shift>0)?(1<<(shift-1)):0; q=...
- Use HVX_Vector / Q6_V* intrinsics for HVX reward bonus.

## Signature
```c
void candidate_kernel(const int8_t *X, const int8_t *W1, const int32_t *b1,
                      const int8_t *W2, const int32_t *b2,
                      const int8_t *gelu_lut,
                      int8_t *out,
                      int32_t mult1, int shift1,
                      int32_t mult2, int shift2);
```
