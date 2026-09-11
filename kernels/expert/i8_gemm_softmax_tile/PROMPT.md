# Task: i8_gemm_softmax_tile

## Overview
Implement `candidate_kernel` in C for the Hexagon HVX DSP.

Compute **GEMM tile** (int8 inputs -> int32 accumulators -> int8 scores) followed by
**rowwise integer softmax** using a runtime exp LUT, producing uint8 output.

## Exact integer formula (MUST match bit-for-bit)

### Step A: GEMM

```
For each i in [0, GM), j in [0, GN):
  acc[i*GN+j] = sum_{k=0}^{GK-1} A[i*GK+k] * B[k*GN+j]   (int32)
```

### Step B: Saturating clamp to int8

```
scores[i*GN+j] = (int8_t) clamp(acc[i*GN+j], -128, 127)
```

Note: this is **saturating clamp**, NOT C truncation cast.
`(int8_t)(int32_t)` truncates (low byte only) -- that is WRONG here.

### Step C: Rowwise softmax

```
For each row i in [0, GM):
  m    = max(scores[i*GN + j])            for j in [0, GN)
  idx  = clamp((int)(scores[i*GN+j] - m), -255, 0) + 255
  e[j] = exp_lut[idx]                    (uint8, runtime, opaque)
  S    = sum(e[j])                        (int32)
  out[i*GN+j] = (uint8)((e[j]*255 + S/2) / S)   (round-half-down)
```

## Constants
- GM=16, GN=16, GK=16 (compile-time via `#define` in kernel_api.h)
- exp_lut is a **runtime** 256-entry uint8 array -- do NOT hardcode any values.

## Function signature
```c
void candidate_kernel(const int8_t *A, const int8_t *B,
                      uint8_t *out,
                      const uint8_t *exp_lut);
```

## HVX guidance
- GEMM inner loop: `Q6_Vw_vrmpyacc_VwVbVb` for 4-way accumulate per cycle.
- Clamp: `Q6_Vb_vsat_VhVh` saturates int16 pairs to int8.
- Softmax max-reduce: scalar loop or partial HVX tree-reduce.
- The 16x16 score tile is 256 bytes (two 128B HVX vectors).
