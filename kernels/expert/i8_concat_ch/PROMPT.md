Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out,
                          int C1, int C2, int H, int W);

Concatenate two tensors along the channel axis (channel-first layout).
  a: (C1, H, W) = (3, 8, 11)
  b: (C2, H, W) = (5, 8, 11)
  out: (C1+C2, H, W) = (8, 8, 11)

EXACT index mapping:
  out[c * H*W + h*W + w] = a[c * H*W + h*W + w]          for c in [0, C1)
  out[c * H*W + h*W + w] = b[(c-C1) * H*W + h*W + w]     for c in [C1, C1+C2)

Equivalently: copy a's channel planes first, then b's channel planes.
W=11 is not a multiple of 128; handle the tail. Pure data movement, no arithmetic.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
