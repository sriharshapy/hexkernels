Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *acc, const int32_t *bias, int32_t *out, int R, int C);
`acc` and `out` are [R x C] row-major int32 tiles. `bias` is an int32 array of
length C, ONE value per COLUMN c, BROADCAST across every row r (a genuine 2D
layout -- not a flat 1D elementwise add). Compute:
    out[r*C+c] = acc[r*C+c] + bias[c]     for all r in [0,R), c in [0,C)
Plain int32 addition, no saturation (test inputs never overflow int32). Pinned
shapes: R=20, C=100 -- C is not a multiple of 32 int32 lanes, so handle the tail
on every row. Use HVX intrinsics; include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
