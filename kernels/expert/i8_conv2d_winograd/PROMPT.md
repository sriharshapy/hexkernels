Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, const int8_t *wt, int8_t *out,
                          int H, int W, int C_in, int C_out,
                          int32_t mult, int shift, int8_t zp);
Conv2d 3x3 + requantize, int8->int8.
Layout: NHWC (batch=1). in[H][W][C_in], wt[C_out][3][3][C_in], out[H][W][C_out].
H=12, W=12, C_in=8, C_out=8 (also passed as runtime args). Padding: SAME (1 pixel zero-pad each side), stride=1.

CORRECTNESS is defined by direct 3x3 convolution (NOT Winograd):
  For each output element out[y][x][co]:
    acc = sum_{ky in 0..2, kx in 0..2, ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]
    (zero-pad out-of-bounds positions)
  Requantize to int8 (mult, shift, zp are RUNTIME params -- do NOT hardcode):
    v    = (int64_t)acc * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
    r   += zp
    out[y][x][co] = saturate_to_int8(r)   // clamp to [-128, 127]

OPTIMIZATION HINT: you may implement Winograd F(2,3) tiling for efficiency, but the output
must be bit-exact with the direct-conv reference above (same int32 accumulation, same requantize).

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
