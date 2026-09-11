Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int n, int32_t *out);
Compute out[0] = max over i in [0,n) of |a[i]| (a[] is int8). Note |-128| = 128, so the
result does not fit in an int8 and must not be computed with a saturating byte abs.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Sign-extend bytes to halfwords (Q6_Wh_vsxt_Vb) so
|-128|=128 fits, take the halfword abs, vmax-accumulate, then horizontally max the 64 lanes.
Handle any sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
