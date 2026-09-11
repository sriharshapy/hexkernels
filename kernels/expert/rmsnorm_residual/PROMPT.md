# Task: rmsnorm_residual

Implement `candidate_kernel` in C for Hexagon HVX (128-byte vectors).

## Signature
```c
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const uint8_t *inv_lut);
```

## Semantics
Fused RMSNorm-after-residual: `out = RMSNorm(x + residual)` — all int8, NO floating point.
**RMSNorm has NO mean subtraction and NO beta (additive bias).**

**Step 1 — saturating residual add:**
```
t[i] = clamp((int32)x[i] + (int32)residual[i], -128, 127)
```

**Step 2 — RMSNorm on t (pinned integer formula, SHIFT=8):**
```
a. rms2  = (int32)(sum_i t[i]^2) / n            (integer truncation, >= 0)
b. r_idx = clamp(rms2, 0, 255)
c. inv   = inv_lut[r_idx]                        (uint8, runtime input)
d. scaled= (t[i] * (int32)gamma[i] + 64) >> 7   (round-half-up)
e. normed= (scaled * (int32)inv + 128) >> 8      (round-half-up, SHIFT=8)
f. out[i]= clamp(normed, -128, 127)
```

## HVX guidance
- Use `HVX_Vector` (128B = 128 int8 lanes) from `hexagon_types.h`.
- Saturating int8 add: `Q6_Vb_vadd_VbVb`.
- For the scalar pass (rms2 reduction) process all n elements first.
- Handle tail (n=113, not a multiple of 128) carefully.

## Constraints
- `n=113` (NOT a multiple of 128); all pointers 128-byte aligned.
- `gamma` and `inv_lut` are runtime inputs — do NOT hardcode them.
- Integer only (no float, no libm).
- Do NOT add a mean subtraction step (that is LayerNorm, not RMSNorm).
