Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in,
                          const int8_t *dw_wt, const int32_t *dw_bias,
                          const int8_t *pw_wt, const int32_t *pw_bias,
                          int8_t *out,
                          int H, int W, int C_in, int C_out,
                          int32_t dw_mult, int dw_shift, int8_t dw_zp,
                          int32_t pw_mult, int pw_shift, int8_t pw_zp);
Full MobileNet depthwise-separable block (BN-folded) + requantize, int8->int8.
H=8, W=8, C_in=8, C_out=8 (also passed as runtime args). Layout: NHWC (batch=1).
  in[H][W][C_in], dw_wt[C_in][3][3], dw_bias[C_in], pw_wt[C_out][C_in], pw_bias[C_out], out[H][W][C_out].

dw_mult, dw_shift, dw_zp, pw_mult, pw_shift, pw_zp are RUNTIME params -- do NOT hardcode.

--- DW stage (per channel c, SAME zero-padding, stride=1) ---
For each (y, x, c):
  dw_acc = sum_{ky in 0..2, kx in 0..2} in_pad[y+ky-1][x+kx-1][c] * dw_wt[c][ky][kx]
  dw_biased = dw_acc + dw_bias[c]
  dw_relu = max(dw_biased, 0)   // ReLU on int32
  dw_q[y][x][c] = requantize dw_relu to int8:
    v    = (int64_t)dw_relu * dw_mult
    half = dw_shift > 0 ? (1LL << (dw_shift-1)) : 0
    r    = (v >= 0) ? (v+half) >> dw_shift : -(((-v)+half) >> dw_shift)
    r   += dw_zp
    dw_q[y][x][c] = saturate_to_int8(r)

--- PW stage (pointwise 1x1, per output channel co) ---
For each (y, x, co):
  pw_acc = sum_{c} dw_q[y][x][c] * pw_wt[co][c]    // dot product on int8 dw_q
  pw_biased = pw_acc + pw_bias[co]
  pw_relu = max(pw_biased, 0)   // ReLU on int32
  out[y][x][co] = requantize pw_relu to int8 using pw_mult, pw_shift, pw_zp (same formula)

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
