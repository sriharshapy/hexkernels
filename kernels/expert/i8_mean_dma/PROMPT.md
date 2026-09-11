Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int n, int32_t *out);
Compute out[0] = (sum_{i=0}^{n-1} a[i]) / n, a signed int32 sum divided by n with C
integer truncation toward zero (a[] is int8).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). A vrmpy against a splat of 0x01010101 reduces 4 signed
bytes into one int32 lane; horizontally sum the 32 lanes, then divide by n. Handle any
sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
