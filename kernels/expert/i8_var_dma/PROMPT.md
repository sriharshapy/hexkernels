Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int n, int32_t *out);
Compute the integer variance out[0] = (n*sum(a[i]^2) - (sum a[i])^2) / n with integer
floor division (a[] is int8). Accumulate sum(a) and sum(a^2) exactly; all intermediates
fit in int64.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Use two vrmpy accumulators: a against a splat of
0x01010101 for sum(a), and a against itself for sum(a^2); reduce the 32 int32 lanes of each
into int64 and finish the variance formula in int64. Handle any sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
