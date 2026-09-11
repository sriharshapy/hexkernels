# Task: i8_conv1d_depthwise_requant

Implement `candidate_kernel` for per-channel (depthwise) 1D convolution + requantize: int8 -> int8.

## Signature

```c
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int C, int L, int K,
                      int32_t mult, int shift, int8_t zp);
```

## Semantics

Layout is channels-first (planar): channel `c` data starts at `x[c*(L+K-1)]`.
Each channel has its own K taps at `taps[c*K]`. Output is `out[c*L + i]`.

```
acc = sum_{k=0}^{K-1} x[c][i+k] * taps[c][k]    // int32 per-channel accumulation
// requantize (round-half-away-from-zero):
v    = (int64_t)acc * mult
half = shift > 0 ? (1LL << (shift-1)) : 0
r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
r   += zp
out[c][i] = saturate_to_int8(r)
```

`x` has `C * (L + K - 1)` elements (VALID: no zero-pad, caller provides extra samples).
Output has `C * L` elements.

## Parameters (harness values)
- C=16, L=256, K=7

## Anti-cheat
Harness sweeps 3 different (mult, shift, zp) sets at runtime. Do NOT hardcode them.

## HVX guidance

Process each channel independently. Within a channel, use HVX vectors to accumulate
K multiply-add terms. `Q6_Vw_vmpyacc_VwVhVh` for int16->int32 partial sums.
Requantize using vmpy + vshr with appropriate rounding.
