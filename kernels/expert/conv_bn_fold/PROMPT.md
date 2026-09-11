# Task: conv_bn_fold

Implement `candidate_kernel` from `kernel_api.h`:
**conv2d (3x3) with folded batchnorm** (per-channel scale+shift) + requantize -> int8.

## Signature
```c
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      const int32_t *bias,
                      const int32_t *bn_scale, const int *bn_shift,
                      int8_t zp,
                      int8_t *out,
                      int H, int W, int C_in, int C_out);
```

## Layout (NHWC)
- `in`      : `[H][W][C_in]`         int8
- `wt`      : `[C_out][3][3][C_in]`  int8 (folded bn weights)
- `bias`    : `[C_out]`              int32 (folded bn offset per output channel)
- `bn_scale`: `[C_out]`              int32 per-channel multiplier
- `bn_shift`: `[C_out]`              int   per-channel shift
- `zp`      : global int8 zero-point
- `out`     : `[H][W][C_out]`        int8

## Computation per out[y][x][co]
1. Convolution accumulate (int32):
   ```
   acc = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]
   ```
   SAME padding: out-of-bounds -> 0.

2. Add folded bias:
   ```
   biased = acc + bias[co]
   ```

3. Per-channel requantize (round-half-away-from-zero):
   ```
   v    = (int64_t)biased * bn_scale[co]
   half = bn_shift[co] > 0 ? (1LL << (bn_shift[co]-1)) : 0
   r    = (v >= 0) ? (v+half)>>bn_shift[co] : -(((-v)+half)>>bn_shift[co])
   r   += zp
   out[y][x][co] = saturate_to_int8(r)
   ```

`bn_scale[]`, `bn_shift[]`, and `zp` are **runtime** -- do NOT hardcode them.

## HVX hints
- Vectorize across the HW spatial dimension.
- Per-channel scale/shift differ per `co` -- load them per output-channel iteration.
- `Q6_Vw_vmpyacc_VwVbVb` for int8*int8->int32 inner product.
- Use int64 accumulation for the scale multiplication to avoid overflow.
