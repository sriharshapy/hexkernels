Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *a, const int16_t *b, int n, int32_t *out);
Compute out[0] = sum over i in [0,n) of a[i]*b[i], accumulated in signed int32.
Inputs are int16 bounded to [-16,16] so the exact sum fits int32 without overflow.

n is large (working set for the two int16 inputs exceeds L2), so the task is
DDR-bandwidth-bound. To go fast, DMA tiles of BOTH a[] and b[] from DDR into VTCM and run
the dot-product reduction on the fast on-chip copies, double-buffering so the next tile
pair's DMA overlaps the current tile's reduce. VTCM is identity-mapped at 0xd8400000. uDMA:
static/global Type-0 descriptors {next,ctrl=len_bytes,src,dst}; address the int16 tiles in
BYTES. A widening int16 multiply (Q6_Ww_vmpy_VhVh) yields a pair of int32 products; add the
lo/hi halves into two int32 accumulators and horizontally sum at the end. Handle any tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
