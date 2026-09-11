Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *in, uint8_t *out, int n);
Compute an inclusive binary threshold with fixed threshold 128:
 out[i] = (in[i] >= 128) ? 255 : 0.
in and out are uint8. The comparison is INCLUSIVE: in[i]==128 -> 255.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of in[] into VTCM with the uDMA engine and compute
on the on-chip copy, double-buffering so the next tile's DMA overlaps the current tile's
compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)` /
`Q6_R_dmwait()`. The descriptor MUST be static/global (a stack descriptor no-ops at -O2).
n is a multiple of 128; handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
