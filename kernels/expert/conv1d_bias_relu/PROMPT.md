# Task: conv1d_bias_relu

Implement `candidate_kernel` for a fused 1D convolution + bias + ReLU -> int8.

## Signature

```c
void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                      int8_t *out,
                      int L, int K, int C);
```

## Semantics

Layout is channels-last (interleaved): `x[i][c]` at `x[i*C + c]`, same for `out`.
Taps are `taps[k][c]` at `taps[k*C + c]` -- one independent filter per channel.

```
acc = sum_{k=0}^{K-1} x[i+k][c] * taps[k][c]   // int32 accumulation
biased = acc + bias[c]
out[i][c] = sat8(max(biased, 0))                 // ReLU then saturate to int8
```

`x` has `(L + K - 1) * C` elements (VALID convolution, no zero-padding needed).
Output has `L * C` elements.

## Parameters (harness values)
- L=512, K=5, C=16

## HVX guidance

Use `HVX_Vector` / `Q6_Vb_vadd_VbVb` / `Q6_Vh_vmpy_VbVb` or similar intrinsics.
Process 128 bytes (one HVX vector) at a time. For a channels-last layout with
C=16, each vector holds 8 output positions. Accumulate in int32 with
`Q6_Vw_vmpyacc_VwVhVh` or multi-step multiply-accumulate.

Do NOT hardcode K, L, or C -- read runtime args.
