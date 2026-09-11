Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, int8_t *out, int n);

Compute out[i] = (x[i] > 0) ? x[i] : 0 for i in [0, n) (signed int8 ReLU).

This is a small IN-CACHE tile epilogue (e.g. immediately after a conv/matmul
tile is already resident in L1/L2, unlike the large-N DDR-bound streaming
ReLU variant) -- n=1000 (NOT a multiple of 128; handle the tail path). No
DMA or l2fetch is needed or expected here; this is pure HVX vector compute.
Edge cases include an all-negative block, an all-positive block, exact zero
(relu(0)=0), and x=-128 (the most negative int8 -- must clamp to 0, not
wrap).

Implement ONLY this function (Hexagon HVX C). Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main(). Respond with a single complete C code block.
