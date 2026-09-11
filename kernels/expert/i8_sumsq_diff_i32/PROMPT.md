Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out);
Compute out[0] = sum of (a[i] - b[i])^2 for i in [0, n) — the L2-distance-squared.
Inputs are signed int8; the result fits in int32 (n=1024, max |diff|=255, 1024*255^2 < 2^31).
n is 1024 (multiple of 128).
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
