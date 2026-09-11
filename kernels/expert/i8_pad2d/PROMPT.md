Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out,
                          int H, int W, int P);

Zero-pad a 2D int8 array by P pixels on all four sides.

Pinned shapes: H=10, W=13, P=3.
  Output size: (H+2*P) x (W+2*P) = 16 x 19.

EXACT mapping:
  For output pixel (r, c) where r in [0, H+2P), c in [0, W+2P):
    if P <= r < H+P AND P <= c < W+P:
        out[r * (W+2*P) + c] = in[(r-P) * W + (c-P)]
    else:
        out[r * (W+2*P) + c] = 0

Border pixels must be ZERO (not edge-replicate, not reflect). OW=19 is not a multiple of 128.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
