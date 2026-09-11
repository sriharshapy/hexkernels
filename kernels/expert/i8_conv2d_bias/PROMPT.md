# Task: i8_conv2d_bias

Implement `candidate_kernel` from `kernel_api.h`:
**conv2d (3x3, stride=1, SAME padding) + int32 bias + requantize -> int8**.

## Signature
```c
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
```

## Layout (NHWC)
- `in`  : `[H][W][C_in]`         int8
- `wt`  : `[C_out][3][3][C_in]`  int8
- `bias`: `[C_out]`              int32 (one per output channel)
- `out` : `[H][W][C_out]`        int8

## Computation per out[y][x][co]
1. Accumulate (int32):
   ```
   acc = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]
   ```
   SAME padding: out-of-bounds pixels -> 0.

2. Add bias:
   ```
   biased = (int64_t)acc + (int64_t)bias[co]
   ```

3. Requantize (round-half-away-from-zero):
   ```
   v    = biased * mult
   half = shift > 0 ? (1LL << (shift-1)) : 0
   r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
   r   += zp
   out[y][x][co] = saturate_to_int8(r)
   ```

`mult`, `shift`, `zp` are **runtime** -- do NOT hardcode them.

## HVX hints
- Vectorize across the HW spatial dimension.
- `Q6_Vw_vmpyacc_VwVbVb` for int8*int8->int32 accumulation.
- Bias is int32 -- add before the requantize scale step.
- Watch for int64 overflow when multiplying the biased accumulator by mult.
