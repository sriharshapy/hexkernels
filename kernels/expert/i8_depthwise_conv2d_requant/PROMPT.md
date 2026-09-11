# Task: i8_depthwise_conv2d_requant

Implement `candidate_kernel` from `kernel_api.h`:
a fused **depthwise 2D convolution (3x3, stride=1, SAME padding) + requantize** in int8.

## Signature
```c
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp);
```

## Layout
- `in`  : `[H][W][C]` int8, NHWC
- `wt`  : `[C][3][3]` int8, one 3x3 filter **per channel** (depthwise -- NOT across channels)
- `out` : `[H][W][C]` int8, same spatial size

## Computation
For each output element `out[y][x][c]`:
1. Accumulate (int32):
   ```
   acc = sum_{ky=0..2, kx=0..2}  in_pad[y+ky-1][x+kx-1][c] * wt[c][ky][kx]
   ```
   Out-of-bounds pixels are zero (SAME padding).
2. Requantize (round-half-away-from-zero):
   ```
   v    = (int64_t)acc * mult
   half = shift > 0 ? (1LL << (shift-1)) : 0
   r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
   r   += zp
   out[y][x][c] = saturate_to_int8(r)   // clamp to [-128, 127]
   ```

`mult`, `shift`, `zp` are **runtime parameters** -- do NOT hardcode them.

## HVX hints
- Use `HVX_Vector` (128 bytes = 128 int8 lanes) for vectorizing the spatial loop.
- `Q6_Vb_vadd_VbVb`, `Q6_Vw_vmpyacc_VwVbVb` for accumulation.
- Process C channels together across the HW spatial dimension.
- Handle SAME-padding border pixels separately or with masked loads.
