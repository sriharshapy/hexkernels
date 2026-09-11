Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
Compute out[0] = min over i in [0,n) of a[i] (a[] is uint8).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Initialise the accumulator to all 0xFF and use
Q6_Vub_vmin_VubVub to accumulate 128 lanes, then horizontally min the lanes. Handle any
sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
