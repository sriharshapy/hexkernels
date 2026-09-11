Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int n, int32_t *out);
Set out[0] to the index of the minimum element of the int8 array a[0..n-1].
Ties are broken by the LOWEST (first) index.

n is large (working set exceeds L2), so finding the min value is DDR-bandwidth-
bound. To go fast, DMA tiles of a[] from DDR into VTCM and run the min-value
reduction (byte vmin) on the fast on-chip copies, double-buffering so the next
tile's DMA overlaps the current tile's reduce; then locate the first index whose
value equals the min with an early-exit HVX vcmp_eq scan. VTCM is identity-mapped
at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
Q6_dmstart_A(&d)/Q6_R_dmwait().

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT
write main(). Respond with a single complete C code block and CLOSE the fence.
