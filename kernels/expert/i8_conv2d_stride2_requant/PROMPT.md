# Task: i8_conv2d_stride2_requant

Implement `candidate_kernel` from `kernel_api.h`:
**conv2d (3x3, stride=2, SAME padding) + requantize -> int8**.

Stride-2 halves the spatial resolution: input is H x W, output is (H/2) x (W/2).

## Signature
```c
void candidate_kernel(const int8_t *in, const int8_t *wt,
                      int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t mult, int shift, int8_t zp);
```

## Layout (NHWC)
- `in`  : `[H][W][C_in]`               int8 (input spatial dims)
- `wt`  : `[C_out][3][3][C_in]`        int8
- `out` : `[H/2][W/2][C_out]`          int8 (output spatial dims -- H and W are even)

## Computation per out[oy][ox][co]  (oy in [0..H/2), ox in [0..W/2))
```
iy_base = oy * 2    // stride=2
ix_base = ox * 2

acc = sum_{ky=0..2, kx=0..2, ci=0..C_in-1}
        in_pad[iy_base+ky-1][ix_base+kx-1][ci] * wt[co][ky][kx][ci]

SAME padding: in_pad[y][x][c] = in[y][x][c]  if 0<=y<H and 0<=x<W,  else 0

v    = (int64_t)acc * mult
half = shift > 0 ? (1LL << (shift-1)) : 0
r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
r   += zp
out[oy][ox][co] = saturate_to_int8(r)
```

`mult`, `shift`, `zp` are **runtime** -- do NOT hardcode them.

## SAME padding for stride=2 with f=3, even H and W:
- `pad_top = 1`, `pad_left = 1` (the first output pixel samples `in[-1][-1]` which is 0).
- No bottom/right padding needed when H and W are both even.

## HVX hints
- With stride=2, each output pixel is independent -- no overlap after stride.
- Vectorize across the `co` (output channel) or `ox` dimension.
- `Q6_Vw_vmpyacc_VwVbVb` for int8*int8->int32 accumulation.
- Memory access pattern: input is strided (skip every other row/col).
