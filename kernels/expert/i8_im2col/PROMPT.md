Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out,
                          int C, int H, int W,
                          int K, int S,
                          int OH, int OW);

im2col: lower a convolution into a column matrix of receptive fields.

Pinned shapes: C=3, H=8, W=9, K=3 (square kernel), S=1 (stride), P=0 (no padding).
  OH = (H-K)/S+1 = 6,  OW = (W-K)/S+1 = 7.

Output matrix out[C*K*K][OH*OW] = out[27][42], stored ROW-MAJOR.

EXACT index mapping (must match bit-for-bit):
  row = c*K*K + kh*K + kw          (channel major, then kernel rows, then kernel cols)
  col = oh*OW + ow                 (output spatial positions, row-major)
  out[row * OH*OW + col] = in[c*H*W + (oh*S + kh)*W + (ow*S + kw)]

Pure data movement, no arithmetic. OW=7 is not a multiple of 128; handle the tail.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
