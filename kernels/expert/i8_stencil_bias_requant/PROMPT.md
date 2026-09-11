# Task: i8_stencil_bias_requant

Implement a 3x3 stencil convolution + bias + requantize pipeline on int8 data using HVX.

## Signature
```c
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      const int8_t *weights, int32_t bias,
                      int32_t mult, int shift, int8_t zp);
```

## Semantics
For each pixel (y, x):
1. Stencil: `acc = sum_{dy in [-1,1], dx in [-1,1]} weights[(dy+1)*3+(dx+1)] * in[clamp(y+dy)*w + clamp(x+dx)]`
   Border: clamp-to-edge. Accumulate in int32.
2. Bias add: `v = acc + bias`
3. Requantize (round-half-away-from-zero):
   - `half = shift > 0 ? (1LL << (shift-1)) : 0`
   - `r = (v >= 0) ? (v*mult + half) >> shift : -(((-v)*mult + half) >> shift)`
   - `r += zp`
   - `out[y*w+x] = saturate_i8(r)` -- clamp to [-128, 127]

## Parameters (runtime -- do NOT hardcode)
- `weights`: 9 int8 values, row-major 3x3
- `bias`: int32 scalar
- `mult`, `shift`, `zp`: requantization parameters

## HVX hints
- Load 3 rows; widen i8->i16 with `Q6_Vb_vshuffe_VbVb` / `Q6_Vh_vsxt_Vb`.
- Accumulate MAC into i32 vectors using `Q6_Vw_vmpyacc_VwVhRh`.
- Add bias using `Q6_Vw_vadd_VwVw` with a broadcast scalar vector.
- Requantize with shift + saturation to int8.

## Anti-cheat
Three distinct (weights, bias, mult, shift, zp) sets are swept.
A kernel hardcoding any parameters will fail at least 2 of 3 sweeps.
