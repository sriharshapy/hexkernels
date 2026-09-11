Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, uint8_t *out, int n);
Compute out[i] = max(a[i] - 128, 0) for i in [0, n): a zero-point ReLU, implemented as
a uint8 SATURATING subtract of 128 (matching Q6_Vub_vsub_VubVub_sat; floors at 0
instead of wrapping negative).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
double-buffer VTCM for BOTH directions: DMA the next tile of a[] in while computing
the current tile's ReLU, and DMA the current tile's result out, so DDR latency on
both the load and the store is hidden behind compute. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is not necessarily a multiple of the tile size or
of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
