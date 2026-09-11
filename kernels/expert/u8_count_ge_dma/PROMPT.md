Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, int n, uint8_t t, int32_t *out);
Compute out[0] = number of indices i in [0,n) with a[i] >= t (inclusive; a[] is
unsigned uint8). The comparison MUST be inclusive '>=', not strict '>'. t is a
runtime argument (fixed to 100 in this task) — read it, do not hardcode the count.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and count on the fast on-chip copies, double-buffering so
the next tile's DMA overlaps the current tile's count. VTCM is identity-mapped at
0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Build a 0/1 byte mask (unsigned compare + vmux) and
vrmpy it against a splat of 0x01010101 to sum four mask bytes into each int32 lane, then
horizontally sum the 32 lanes. Handle any sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
