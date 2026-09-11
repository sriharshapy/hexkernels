Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out,
                          int C_in, int H_in, int W_in, int b);

Depth-to-space: rearrange channel dimension back into spatial blocks (inverse of space-to-depth).

Pinned shapes: C_in=12, H_in=4, W_in=5, block factor b=2.
  C_out = C_in / (b*b) = 3,  H_out = H_in*b = 8,  W_out = W_in*b = 10.

EXACT index mapping (must match bit-for-bit):
  c_in = c * b*b + bh*b + bw    (c in [0,C_out), bh,bw in [0,b))
  out[c * H_out*W_out + (oh*b+bh)*W_out + (ow*b+bw)] = in[c_in * H_in*W_in + oh*W_in + ow]
  (oh in [0, H_in),  ow in [0, W_in))

W_in=5 is not a multiple of 128; handle the tail. Pure data movement, no arithmetic.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
