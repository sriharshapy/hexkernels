Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, int8_t *out, int n);
For each i in [0,n): out[i] = (x[i] < 0) ? 0 : x[i] (int8 ReLU).

n is large (1B read + 1B write per element => DDR-bandwidth-bound). To go fast
you should hide DDR latency: stage tiles of x[] into VTCM with the uDMA engine
and compute on the on-chip copy, then DMA the result back to DDR. For best
throughput, double-buffer: prefetch the next input tile (async uDMA) while
computing on the current tile, so DMA and compute overlap instead of serializing.
VTCM is identity-mapped at 0xd8400000. uDMA: build static/global Type-0
descriptors {next,ctrl=len,src,dst}, then `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. Descriptors MUST be static/global (a stack descriptor
no-ops at -O2). n is a multiple of 128; handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
