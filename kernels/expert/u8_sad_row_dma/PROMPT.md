Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, const uint8_t *b, int n, int32_t *out);
Compute out[0] = sum over i in [0,n) of |a[i]-b[i]|. Inputs are unsigned bytes; use the
absolute difference (NOT the signed difference). The result fits in int32.

n is large (working set for the two uint8 inputs exceeds L2), so the task is
DDR-bandwidth-bound. To go fast, DMA tiles of BOTH a[] and b[] from DDR into VTCM and run
the SAD reduction on the fast on-chip copies, double-buffering so the next tile pair's DMA
overlaps the current tile's reduce. VTCM is identity-mapped at 0xd8400000. uDMA:
static/global Type-0 descriptors {next,ctrl=len,src,dst} (chain a+b via `next`),
Q6_dmstart_A(&desc)/Q6_R_dmwait(). Q6_Vub_vabsdiff_VubVub gives |a-b| per unsigned byte;
an UNSIGNED 4-byte vrmpy (Q6_Vuw_vrmpyacc_VuwVubVub) against a splat of 1s sums those into
uint32 lanes (|diff| up to 255 stays exact); horizontally sum the 32 lanes. Handle any tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
