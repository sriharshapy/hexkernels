# Task: layernorm_residual

Implement `candidate_kernel` in C for Hexagon HVX (128-byte vectors).

## Signature
```c
void candidate_kernel(const int8_t *x, const int8_t *residual, int8_t *out, int n,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
```

## Semantics
Fused LayerNorm-after-residual: `out = LayerNorm(x + residual)` — all int8, NO floating point.

**Step 1 — saturating residual add:**
```
t[i] = clamp((int32)x[i] + (int32)residual[i], -128, 127)
```

**Step 2 — LayerNorm on t (pinned integer formula, SHIFT=8):**
```
a. mu    = (int32)(sum_i t[i]) / n              (integer truncation toward zero)
b. var   = (int32)(sum_i (t[i]-mu)^2) / n       (integer truncation, >= 0)
c. v_idx = clamp(var, 0, 255)
d. inv   = inv_lut[v_idx]                        (uint8, runtime input)
e. d[i]  = (int32)t[i] - mu
f. scaled= (d[i] * (int32)gamma[i] + 64) >> 7   (round-half-up)
g. normed= (scaled * (int32)inv + 128) >> 8      (round-half-up, SHIFT=8)
h. out[i]= clamp(normed + (int32)beta[i], -128, 127)
```

## HVX guidance
- Use `HVX_Vector` (128B = 128 int8 lanes) from `hexagon_types.h`.
- Useful intrinsics: `Q6_Vb_vadd_VbVb` (saturating int8 add), `Q6_Vw_vmpyacc_VwVhRh` (accumulate).
- For the scalar prefix (mean/variance) pass over all n elements before the per-element HVX loop.
- Handle tail (n=113, not a multiple of 128) carefully.

## Constraints
- `n=113` (NOT a multiple of 128); all pointers 128-byte aligned.
- `gamma`, `beta`, `inv_lut` are runtime inputs — do NOT hardcode them.
- Integer only (no float, no libm).
- `used_hvx` is true only if `HVX_Vector` or `Q6_V*`/`Q6_W*` intrinsics appear in non-comment code.
