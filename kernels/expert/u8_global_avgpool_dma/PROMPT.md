Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
Compute the GLOBAL AVERAGE POOL of a single flat uint8 plane:
    out[0] = (uint8_t)( (a[0]+a[1]+...+a[n-1]) / n )
using an int32 accumulator; the division truncates and the result is cast to
uint8. n is fixed at 1114112 for this task.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). An unsigned vrmpy (Q6_Vuw_vrmpyacc_VuwVubVub) against a
splat of 0x01010101 reduces 4 unsigned bytes into one u32 lane; horizontally sum the 32
lanes, then divide by n and cast to uint8. Handle any sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
