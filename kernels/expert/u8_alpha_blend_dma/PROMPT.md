Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out,
                          int n, uint16_t alpha);
Compute out[i] = (uint8_t)((alpha*a[i] + (256-alpha)*b[i] + 128) >> 8) for i in [0, n).
alpha is a runtime uint16 in [0,256] (256->pure a, 0->pure b) -- do NOT hardcode it.
Note the identity out = b + ((alpha*(a-b)+128) >> 8), which fits int16 halfword lanes.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] and b[] into VTCM with the uDMA engine and
compute on the on-chip copies, double-buffering so the next tiles' DMA overlaps the
current tile's compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000.
uDMA: build static/global Type-0 descriptors {next,ctrl=len,src,dst}, then
Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a stack
descriptor no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.
Widen bytes with Q6_Wuh_vzxt_Vub; pack results with Q6_Vub_vasr_VhVhR_sat.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
