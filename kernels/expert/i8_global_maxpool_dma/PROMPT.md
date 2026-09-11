Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int n, int8_t *out);
Compute the GLOBAL MAX POOL of a single flat int8 plane (SIGNED max):
    out[0] = max(a[0], a[1], ..., a[n-1])   in [-128, 127].
n is fixed at 1114112 for this task.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Accumulate with the SIGNED byte max
Q6_Vb_vmax_VbVb (seed the accumulator to -128), then horizontally max the 128 lanes.
Handle any sub-tile tail. Note: an unsigned-byte max is WRONG for signed data.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
