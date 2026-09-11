Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int n, int32_t *out);
Compute out[0] = sum over i in [0,n) of a[i]*b[i], accumulated in signed int32.
Inputs are bounded to [-16,16] so the exact sum fits int32 without overflow.

n is large (working set for the two int8 inputs exceeds L2), so the task is
DDR-bandwidth-bound. To go fast, DMA tiles of BOTH a[] and b[] from DDR into VTCM and run
the dot-product reduction on the fast on-chip copies, double-buffering so the next tile
pair's DMA overlaps the current tile's reduce. VTCM is identity-mapped at 0xd8400000. uDMA:
static/global Type-0 descriptors {next,ctrl=len,src,dst} (chain a+b with the `next` field),
Q6_dmstart_A(&desc)/Q6_R_dmwait(). vrmpy (Q6_Vw_vrmpyacc_VwVbVb) reduces 4 byte products
a*b into each int32 lane; horizontally sum the 32 lanes at the end. Handle any sub-tile tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
