# Task: i8_depthwise_conv2d_bias_relu

Implement `candidate_kernel` from `kernel_api.h`:
**depthwise conv2d (3x3) + bias + ReLU + requantize -> int8**.

## Signature
```c
void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                      int8_t *out,
                      int H, int W, int C,
                      int32_t mult, int shift, int8_t zp);
```

## Layout (NHWC)
- `in`  : `[H][W][C]`  int8
- `wt`  : `[C][3][3]`  int8, one 3x3 filter **per channel** (depthwise)
- `bias`: `[C]`        int32, per-channel bias
- `out` : `[H][W][C]`  int8, same spatial size

## Computation per out[y][x][c]
1. Depthwise accumulate (int32):
   ```
   acc = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * wt[c][ky][kx]
   ```
   SAME padding: out-of-bounds pixels -> 0.

2. Add bias:
   ```
   biased = (int64_t)acc + (int64_t)bias[c]
   ```

3. ReLU (clamp to zero):
   ```
   after_relu = max(biased, 0)
   ```

4. Requantize (round-half-away-from-zero):
   ```
   v    = after_relu * mult
   half = shift > 0 ? (1LL << (shift-1)) : 0
   r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
   r   += zp
   out[y][x][c] = saturate_to_int8(r)
   ```

`mult`, `shift`, `zp` are **runtime** -- do NOT hardcode them.

**Important**: ReLU is applied to `biased` (after bias), not to the raw accumulator.
A very negative bias can force an entire channel to zero output.

## HVX hints
- Depthwise: each output channel is independent -- no cross-channel accumulation.
- `Q6_Vb_vmax_VbVb` or conditional for the ReLU step.
- Process in 128-byte HVX vectors across HW spatial positions.
