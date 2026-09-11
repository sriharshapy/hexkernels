# Task: i8_conv1d_dilated_requant

Implement `candidate_kernel` for a dilated 1D convolution + requantize: int8 -> int8.

## Signature

```c
void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int dilation,
                      int32_t mult, int shift, int8_t zp);
```

## Semantics

Single-channel VALID dilated FIR. Caller provides `(n + (ntaps-1)*dilation)` input samples.

```
acc = sum_{k=0}^{ntaps-1} x[i + k*dilation] * taps[k]   // int32, taps NOT reversed
// requantize (round-half-away-from-zero):
v    = (int64_t)acc * mult
half = shift > 0 ? (1LL << (shift-1)) : 0
r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
r   += zp
out[i] = saturate_to_int8(r)
```

`dilation >= 1`: with `dilation=1` this is standard conv1d; higher dilation widens receptive field
by sampling every `dilation`-th input element.

## Parameters (harness values)
- n=512, ntaps=7, dilation swept {1, 2, 3}

## Anti-cheat
Harness sweeps 3 dilation values x 2 quant sets = 6 combinations.
Do NOT hardcode dilation, mult, shift, or zp.

## HVX guidance

With `dilation > 1`, sequential output positions access non-contiguous input bytes, making
vectorization harder. Gather the dilated taps with `vmem` or deinterleave intrinsics.
For `dilation=1` the inner loop is a contiguous sliding window -- vectorize that path.
Use `Q6_Vw_vmpyacc_VwVhRh` for int32 accumulation; apply requant as vector multiply+shift.
