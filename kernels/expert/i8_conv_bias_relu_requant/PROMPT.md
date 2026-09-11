Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, const int8_t *wt, const int32_t *bias,
                          int8_t *out,
                          int H, int W, int C_in, int C_out,
                          int32_t mult, int shift, int8_t zp);
Fused 2D conv (3x3, SAME padding, stride=1) + bias + ReLU + requantize, int8->int8.
Layout: NHWC (batch=1, not in shape). in[H][W][C_in], wt[C_out][3][3][C_in], bias[C_out], out[H][W][C_out].
For each output element out[y][x][co]:
  Step 1 -- conv:   acc = sum_{ky,kx,ci} in_pad[y+ky-1][x+kx-1][ci] * wt[co][ky][kx][ci]
                    (zero-pad out-of-bounds; ky,kx in [0,2])
  Step 2 -- bias:   biased = acc + bias[co]
  Step 3 -- relu:   after_relu = max(biased, 0)
  Step 4 -- requantize to int8:
    v    = (int64_t)after_relu * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)  // round half away from 0
    r   += zp
    out[y][x][co] = saturate_to_int8(r)  // clamp to [-128, 127]
H=16, W=16, C_in=8, C_out=8 (also passed as runtime args). mult, shift, zp are runtime params (do NOT hardcode).
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
