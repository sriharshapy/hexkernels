Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n);
Compute out[i] = a[i] for i in [0, n) (an exact tile-staged copy).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
stage each tile of a[] from DDR into VTCM with the uDMA engine (double-buffering so the
next tile's DMA-in overlaps the current tile's store), then write the on-chip tile to
out with a plain HVX vector store. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc) /
Q6_R_dmwait(). The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is not necessarily a multiple of the tile size or of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
