# Task: i8_fir_bias_requant

Implement `candidate_kernel` for a fused FIR filter + scalar bias + requantize: int8 -> int8.

## Signature

```c
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t bias,
 int8_t *out,
 int n, int ntaps,
 int32_t mult, int shift, int8_t zp);
```

## Semantics

Single-channel VALID FIR (no zero-padding). Caller provides `(n + ntaps - 1)` input samples.

```
acc = sum_{k=0}^{ntaps-1} x[i+k] * taps[k] // int32 FIR accumulation (taps NOT reversed)
biased = acc + bias
// requantize (round-half-away-from-zero):
v = (int64_t)biased * mult
half = shift > 0 ? (1LL << (shift-1)) : 0
r = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
r += zp
out[i] = saturate_to_int8(r)
```

`x` has `n + ntaps - 1` elements (no zero-padding needed).
Output has `n` elements.

## Parameters (harness values)
- n=512, ntaps=16

## Anti-cheat
Harness sweeps 3 different (mult, shift, zp) sets. bias=500 (non-trivial). Do NOT hardcode.

## HVX guidance

With ntaps=16 and int8 data, process 128 input bytes = 128 output positions per vector load.
Use `Q6_Ww_vmpyacc_WwVhVh` or dot-product intrinsics for the inner tap loop.
Accumulate partial sums in int32 vectors, then apply requantization as a vector multiply+shift.
