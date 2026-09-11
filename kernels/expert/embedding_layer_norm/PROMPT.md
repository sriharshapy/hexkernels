# Task: embedding_layer_norm

Implement `candidate_kernel` in C for Hexagon HVX (128-byte vectors).

## Signature
```c
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut);
```

## Semantics
Fused embedding gather + LayerNorm: for each token i, gather its embedding row then normalize it.
All int8, NO floating point.

**Per-token operation:**
```
Step 1 — Gather:
  row = table[idx[i], :]   (copy D int8 bytes from vocab table)

Step 2 — LayerNorm on row (pinned formula, SHIFT=8):
  a. mu    = (int32)(sum_j row[j]) / D               (truncation toward zero)
  b. var   = (int32)(sum_j (row[j]-mu)^2) / D        (truncation, >= 0)
  c. v_idx = clamp(var, 0, 255)
  d. inv   = inv_lut[v_idx]                           (uint8, runtime)
  e. d[j]  = (int32)row[j] - mu
  f. scaled= (d[j] * (int32)gamma[j] + 64) >> 7      (round-half-up)
  g. normed= (scaled * (int32)inv + 128) >> 8         (round-half-up, SHIFT=8)
  h. out[i*D + j] = clamp(normed + (int32)beta[j], -128, 127)
```

Each token has its **own** mu, var, inv (computed from its gathered row).
`gamma`, `beta`, `inv_lut` are **shared** across tokens (runtime inputs).

**Pinned dimensions:** `T=32 tokens, D=64, VOCAB=256`.

## HVX guidance
- Use `HVX_Vector` (128B = 128 int8 lanes).
- D=64 = half an HVX vector; load with a masked/partial load or process 2 tokens at once.
- The gather is a memcpy of D=64 bytes per token (single HVX load if aligned).
- Statistics (mean, variance) over D=64 elements: scalar reduction.
- Apply the per-dim formula with a partial HVX vector.

## Constraints
- `T=32`, `D=64`, `VOCAB=256`; all pointers 128-byte aligned.
- `gamma`, `beta`, `inv_lut` are runtime inputs — do NOT hardcode them.
- LayerNorm is per-token (each token row has its own statistics), NOT global.
- Integer only (no float, no libm).
