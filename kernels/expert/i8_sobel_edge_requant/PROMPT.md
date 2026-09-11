# Task: i8_sobel_edge_requant

Implement Sobel gradient magnitude computation followed by requantization to uint8, using HVX.

## Signature
```c
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h,
                      int mult, int shift, int zp);
```

## Semantics
For each pixel (y, x):
1. Compute Sobel gradients with clamp-to-edge borders:
   - `Gx = sum_{dy,dx} KX[dy+1][dx+1] * in[clamp(y+dy)*w + clamp(x+dx)]`
   - `Gy = sum_{dy,dx} KY[dy+1][dx+1] * in[clamp(y+dy)*w + clamp(x+dx)]`
   - KX = [[-1,0,1],[-2,0,2],[-1,0,1]], KY = [[-1,-2,-1],[0,0,0],[1,2,1]]
2. Magnitude: `mag = |Gx| + |Gy|` (L1 norm, range [0, 1020])
3. Requantize (round-half-up, mag >= 0):
   - `half = shift > 0 ? (1 << (shift-1)) : 0`
   - `v = (mag * mult + half) >> shift`
   - `v += zp`
   - `out[y*w+x] = clamp(v, 0, 255)`

## Parameters (runtime -- do NOT hardcode)
- `mult`: integer multiplier
- `shift`: right-shift amount
- `zp`: zero-point offset added after shift

## HVX hints
- Load 3 rows into HVX vectors; compute Gx/Gy as sums of shifted rows.
- `Q6_Vw_vmpyacc_VwVhRh` for weighted row accumulation.
- Absolute value: `Q6_Vw_vabs_Vw`.
- Requantize the i32 magnitude vector using shift + saturation.

## Anti-cheat
Three distinct (mult, shift, zp) sets are swept at runtime.
A kernel that hardcodes any parameter set will fail two out of three sweeps.
