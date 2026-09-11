Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *w, int n, int32_t *out);
Compute out[0] = sum of w[i] * a[i] for i in [0, n) — a weighted dot product.
Both a (data) and w (weights) are signed int8. Result fits in int32.
n is 1024 (multiple of 128).
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
