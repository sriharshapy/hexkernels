Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                          int n, int ntaps, int stride);
Compute a 1D strided FIR as CORRELATION (taps are NOT reversed):
    out[i] = sum over j=0..ntaps-1 of x[i*stride + j] * taps[j],   for i in [0, n).
x has n*stride + ntaps - 1 samples (VALID window, no padding). out has n int32 results.
stride is passed at runtime (pinned to 2 in the harness). ntaps is 8. n is 501 (not a
multiple of 128 — handle the tail). Use a 32-bit signed accumulator. Use HVX intrinsics;
include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
