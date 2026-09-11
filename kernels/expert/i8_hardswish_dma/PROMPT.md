Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out, int n);
For each i in [0,n): relu6 = clamp(x[i] + 3, 0, 6); out[i] = (int8_t)clamp(x[i]*relu6/6,
-128, 127). The division by 6 is C integer division (TRUNCATES toward zero, not floor).
No runtime parameters.

n is large (int8 read + int8 write => DDR-bandwidth-bound). To go fast you should hide
DDR latency: stage tiles of x[] into VTCM with the uDMA engine and compute on the on-chip
copies, double-buffering so the next tile's DMA overlaps the current tile's compute; DMA
results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global
Type-0 descriptor {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. The
descriptor MUST be static/global (a stack descriptor no-ops at -O2). x*relu6 fits int16
(compute in halfword lanes); implement /6 truncation toward zero with a magic multiply in
word lanes (e.g. floor(|p|/6) = (|p|*10923)>>16) and restore the sign. n is a multiple of
128; handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
