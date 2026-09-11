# Task: i8_chunk_norm

## Overview
Implement `candidate_kernel` in C for the Hexagon HVX DSP.

Compute **per-chunk (group) normalization** over a length-`n` int8 vector,
producing int8 output.  The vector is split into `G` equal-size chunks;
each chunk is normalized independently.

## Exact integer formula (MUST match bit-for-bit)

For each chunk `c` in `[0, G)`, let `xc[j] = x[c*chunk + j]`, `chunk = n/G`:

```
Step 1 -- Per-chunk mean (integer truncation toward zero):
  mu_c = (int32)(sum_j xc[j]) / chunk

Step 2 -- Per-chunk variance (integer truncation):
  var_c = (int32)(sum_j (xc[j] - mu_c)^2) / chunk

Step 3 -- LUT index:
  v_idx = (int) clamp(var_c, 0, 255)

Step 4 -- Inverse std lookup:
  inv_c = inv_lut[v_idx]         // 256-entry uint8 runtime input (opaque)

Step 5 -- Per-element normalize + affine:
  For each j in [0, chunk):
    d[j]    = (int32)(xc[j] - mu_c)
    scaled  = (d[j] * (int32)gamma[c*chunk+j] + 64) >> 7   // round-half-up
    normed  = (scaled * (int32)inv_c + 128) >> 8            // round-half-up, SHIFT=8
    out[c*chunk+j] = (int8_t) clamp(normed + (int32)beta[c*chunk+j], -128, 127)
```

## Key points
- `gamma` and `beta` are per-element int8 arrays (length n).
- `inv_lut` is **opaque at compile time** -- always read from the runtime pointer.
- G and n are runtime params; n=128, G=4 -> chunk=32 during evaluation.
- Integer only: no `float`, `double`, or `sqrt()`.
- Each chunk is independent -- do not share mean/var across chunks.

## Function signature
```c
void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
```

## HVX guidance
- Each 32-element chunk fits in one 128B HVX_Vector (32 x int8 = 32 bytes, padded).
- Use `Q6_Vw_vrmpyacc_VwVbVb` or a widening multiply for the variance accumulation.
- For mean reduction: `Q6_R_vextract_VR` extracts a lane; tree-reduce or scalar loop.
- The normalize step (gamma scale + inv_lut scale) maps well to HVX multiply-shift.
