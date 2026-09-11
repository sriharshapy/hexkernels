# Task: matmul_softmax_tile

## Overview
Implement `candidate_kernel` in C for the Hexagon HVX DSP.

Compute a **Q*K^T attention score tile** (int8 inputs -> scaled int8 scores)
followed by **rowwise softmax** (using a runtime exp LUT) producing uint8 output.

## Exact integer formula (MUST match bit-for-bit)

### Step A: Scaled QK^T dot products (-> int8 scores)

For each query row `i` in `[0, SEQ_Q)` and key row `j` in `[0, SEQ_K)`:
```
raw[i*SEQ_K+j] = sum_{d=0}^{HEAD_DIM-1} Q[i*HEAD_DIM+d] * K[j*HEAD_DIM+d]   (int32)

v    = (int64_t)raw * (int64_t)inv_sqrt_d
half = (shift > 0) ? (1LL << (shift-1)) : 0
sc   = (v >= 0) ? (v + half) >> shift : -((-v + half) >> shift)
scores[i*SEQ_K+j] = (int8_t) clamp(sc, -128, 127)
```

### Step B: Rowwise integer softmax

For each row `i` in `[0, SEQ_Q)`:
```
m    = max(scores[i*SEQ_K + j])             for j in [0, SEQ_K)
idx  = clamp((int)(scores[i*SEQ_K+j] - m), -255, 0) + 255
e[j] = exp_lut[idx]                         (uint8, runtime, opaque)
S    = sum(e[j])                             (int32)
out[i*SEQ_K+j] = (uint8)((e[j]*255 + S/2) / S)   (round-half-down)
```

## Constants
- SEQ_Q=16, SEQ_K=16, HEAD_DIM=32 (compile-time via `#define` in kernel_api.h)
- inv_sqrt_d and shift are **runtime params** -- do NOT hardcode them.
- exp_lut is **runtime** -- do NOT hardcode any table values.

## Function signature
```c
void candidate_kernel(const int8_t *Q, const int8_t *K,
                      uint8_t *out,
                      const uint8_t *exp_lut,
                      int32_t inv_sqrt_d, int shift);
```

## HVX guidance
- For the matmul: `Q6_Vw_vrmpyacc_VwVbVb` (4-element SIMD dot product accumulate).
- For the softmax: standard max-subtract -> LUT-lookup -> normalize.
- Intermediate scores fit in a 16x16 int8 tile (256 bytes, two HVX_Vectors).
- Use `int64_t` for the scaling multiply to avoid overflow.
