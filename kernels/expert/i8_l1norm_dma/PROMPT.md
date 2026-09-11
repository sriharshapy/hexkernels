Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *a, int n, int32_t *out);
Compute out[0] = sum_{i=0}^{n-1} |a[i]| as a signed int32 accumulator (a[] is int8).
Note |-128| = 128, so byte-domain abs (which saturates -128 to 127) is incorrect.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, DMA
tiles of a[] from DDR into VTCM and run the reduction on the fast on-chip copies,
double-buffering so the next tile's DMA overlaps the current tile's reduce. VTCM is
identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor {next,ctrl=len,src,dst},
`Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. One exact trick: |a| = a*sign(a); build a per-byte sign
vector (-1 where a<0 else +1) with a compare + vmux and vrmpy a against it so each product
lands exactly in the int32 lanes; horizontally sum the 32 lanes. Handle any sub-tile tail.

Respond with a single complete C code block and CLOSE the fence with ```.
