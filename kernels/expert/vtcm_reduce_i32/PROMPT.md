Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int n, int32_t *out);
Compute out[0] = sum of a[i] for i in [0, n) (exact int32 sum; the reference uses
ordinary 32-bit wraparound addition, order-independent).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
DMA-stage tiles of a[] from DDR into VTCM (double-buffering so the next tile's load
overlaps the current tile's reduction), accumulate each tile with HVX vector add
(Q6_Vw_vadd_VwVw) into a running 32-lane vector accumulator, and horizontally sum the
32 lanes to a scalar only once at the end. VTCM is identity-mapped at 0xd8400000.
uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst}, then
Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a stack
descriptor no-ops at -O2). n is not necessarily a multiple of the tile size or of 32;
handle any remainder with scalar accumulation.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
