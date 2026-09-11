# Task: depthwise_pointwise_block

Implement `candidate_kernel` from `kernel_api.h`:
a fused **MobileNet-style depthwise-separable block** (depthwise 3x3 then pointwise 1x1)
with independent requantize at each stage.

## Signature
```c
void candidate_kernel(const int8_t *in,
                      const int8_t *dw_wt,
                      const int8_t *pw_wt,
                      int8_t *mid, int8_t *out,
                      int H, int W, int C_in, int C_out,
                      int32_t dw_mult, int dw_shift, int8_t dw_zp,
                      int32_t pw_mult, int pw_shift, int8_t pw_zp);
```

## Layout
- `in`    : `[H][W][C_in]`   int8, NHWC
- `dw_wt` : `[C_in][3][3]`   int8, one 3x3 filter per channel (depthwise)
- `pw_wt` : `[C_out][C_in]`  int8, 1x1 pointwise weights
- `mid`   : `[H][W][C_in]`   int8 scratch (write intermediate depthwise output here)
- `out`   : `[H][W][C_out]`  int8 final output

## Computation

### Stage 1: Depthwise conv + requant -> mid
```
For y,x,c:
  acc = sum_{ky,kx} in_pad[y+ky-1][x+kx-1][c] * dw_wt[c][ky][kx]
  mid[y][x][c] = requant(acc, dw_mult, dw_shift, dw_zp)
```

### Stage 2: Pointwise conv + requant -> out
```
For y,x,co:
  acc = sum_{ci} mid[y][x][ci] * pw_wt[co][ci]
  out[y][x][co] = requant(acc, pw_mult, pw_shift, pw_zp)
```

### requant(acc, mult, shift, zp):
```
v    = (int64_t)acc * mult
half = shift > 0 ? (1LL << (shift-1)) : 0
r    = (v >= 0) ? (v+half)>>shift : -(((-v)+half)>>shift)
r   += zp
return saturate_to_int8(r)
```

SAME padding: out-of-bounds pixels are zero.
All mult/shift/zp parameters are **runtime** -- do NOT hardcode them.

## HVX hints
- Process spatial pixels in 128-byte HVX vectors.
- `Q6_Vw_vmpyacc_VwVbVb` for int8*int8->int32 accumulation.
- Two separate requant passes: one for depthwise output, one for pointwise output.
