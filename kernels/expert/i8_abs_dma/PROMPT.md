Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n);
Compute the saturating int8 absolute value: out[i] = |a[i]|, with the single special
case a[i] == -128 saturating to 127 (matches Q6_Vb_vabs_Vb). All other values are
symmetric.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] into VTCM with the uDMA engine and compute
on the on-chip copy, double-buffering so the next tile's DMA overlaps the current tile's
compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc) /
Q6_R_dmwait(). The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is a multiple of 128; handle any sub-tile remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
