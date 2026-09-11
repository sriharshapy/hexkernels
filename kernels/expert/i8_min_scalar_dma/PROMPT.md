Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *in, int8_t *out, int n, int8_t c);
Compute out[i] = min(in[i], c) for i in [0, n), signed int8 (c=0 is a ceiling-at-zero
clamp). c is a runtime scalar -- do NOT hardcode it.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of in[] into VTCM with the uDMA engine and compute
on the on-chip copies, double-buffering so the next tile's DMA overlaps the current
tile's compute; DMA results back to DDR. VTCM is identity-mapped at 0xd8400000.
uDMA: build a static/global Type-0 descriptor {next,ctrl=len,src,dst}, then
`Q6_dmstart_A(&desc)` / `Q6_R_dmwait()`. The descriptor MUST be static/global (a stack
descriptor no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.
A signed byte min is `Q6_Vb_vmin_VbVb`; splat c to all bytes.

Respond with a single complete C code block and CLOSE the fence with ```.
