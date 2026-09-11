Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
For each i in [0,n): out[i] = sat_i8(a[i] + b[i]) (int8 saturating add, clamped
to [-128,127]).

n is large (2B read + 1B write per element => DDR-bandwidth-bound). To go fast
you should hide DDR latency: stage tiles of a[] and b[] into VTCM with the
uDMA engine and compute on the on-chip copies, then DMA the result back to
DDR. For best throughput, double-buffer BOTH input streams: prefetch the next
tile of a[] and the next tile of b[] (async uDMA) while computing on the
current tiles, so DMA and compute overlap instead of serializing. VTCM is
identity-mapped at 0xd8400000. uDMA: build static/global Type-0 descriptors
{next,ctrl=len,src,dst} (one per in-flight transfer), then `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()`. Each descriptor MUST be
static/global (a stack descriptor no-ops at -O2). n is a multiple of 128;
handle any sub-tile remainder.

Respond with a single complete C code block and CLOSE the fence with ```.
