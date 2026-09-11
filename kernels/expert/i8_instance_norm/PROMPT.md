# Task: i8_instance_norm

Implement `candidate_kernel` in C for Hexagon HVX (128-byte vectors).

## Signature
```c
void candidate_kernel(const int8_t *x, int8_t *out,
                      int H, int W, int C,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
```

## Semantics
Instance Normalization: normalize each channel's spatial map independently, int8 in/out, NO floating point.

**Layout:** `x[c * HW + s]` where `HW = H * W`. Each channel `c` is normalized over its `HW` spatial positions.

**Per-channel formula (pinned, SHIFT=8):**
```
count = HW

1. mu    = sum_s x[c*HW+s] / count                   (truncation toward zero)
2. var   = sum_s (x[c*HW+s] - mu)^2 / count          (truncation, >= 0)
3. v_idx = clamp(var, 0, 255)
4. inv   = inv_lut[c * 256 + v_idx]                  (uint8, runtime; per-channel LUT)
5. For each spatial position s in [0, HW):
   a. d      = (int32)x[c*HW + s] - mu
   b. scaled = (d * (int32)gamma[c] + 64) >> 7         (round-half-up)
   c. normed = (scaled * (int32)inv + 128) >> 8        (round-half-up, SHIFT=8)
   d. out[c*HW + s] = clamp(normed + (int32)beta[c], -128, 127)
```

**Pinned dimensions:** `H=16, W=16, C=16, HW=256, n=4096`.

## HVX guidance
- Use `HVX_Vector` (128B = 128 int8 lanes).
- Per-channel spatial loop (HW=256 = 2 HVX vectors): excellent for vectorization.
- Statistics (mean, variance) require a scalar reduction over HW=256 elements per channel.
- `inv_lut` has `C*256=4096` bytes; channel `c` uses `inv_lut + c*256`.

## Constraints
- `gamma`, `beta`, `inv_lut` are runtime inputs — do NOT hardcode any of them.
- Each channel uses its OWN LUT slice (not a shared LUT).
- Integer only (no float, no libm).
- Normalize per-channel (NOT globally, NOT per group).
