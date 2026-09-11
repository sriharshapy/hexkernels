Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);

Compute a 2x2 MAX pool with stride 2 and ZERO-PADDING ("same" padding).
Output dimensions: ow = (w+1)/2, oh = (h+1)/2, stored row-major.
Padding policy: ZERO-PAD — out-of-bounds input positions are treated as 0.
Semantics:
    For each output pixel (ox, oy):
        r = 2*oy, c = 2*ox
        read in[r][c], in[r][c+1], in[r+1][c], in[r+1][c+1]
        (out-of-bounds = 0, NOT replicated border)
        out[oy*ow+ox] = max of those four values
w=129, h=97 (odd dims) → ow=65, oh=49; ow=65 NOT a multiple of 128 — handle the tail.
The last output column (when w is odd) and last output row (when h is odd) each
have a 2x2 window that extends one pixel beyond the input edge — treat those as 0.
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
