Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                          int n, int ntaps);
Compute a 1D FIR with SAME zero-padding as CORRELATION (taps are NOT reversed):
    pad_left = (ntaps-1)/2    (ntaps is odd so centering is exact)
    out[i] = sum over j=0..ntaps-1 of
               (i - pad_left + j >= 0 && i - pad_left + j < n
                ? x[i - pad_left + j] : 0) * taps[j],   for i in [0, n).
x has n samples (NOT extended — boundary must be zero-filled). out has n int32 results.
ntaps is 7 (odd, centered padding: pad_left=3, pad_right=3). n is 1000 (not a multiple
of 128 — handle the tail). The first and last 3 outputs partially overlap the zero-pad
region. Use a 32-bit signed accumulator. Use HVX intrinsics; include <hexagon_types.h>
and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
