Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W);

Convert a tensor from NCHW to NHWC layout (N=1, so N dimension is omitted from the pointer).
    in  is laid out as [C][H][W]:  in[c*H*W + h*W + w]
    out is laid out as [H][W][C]:  out[h*W*C + w*C + c]

Pinned shapes: C=4, H=10, W=13 (W is not a multiple of 128 — handle the tail).
Pure data movement, no arithmetic. No loop fusion that changes the index formula.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
