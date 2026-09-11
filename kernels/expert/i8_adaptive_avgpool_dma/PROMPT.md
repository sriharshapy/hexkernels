Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, int8_t *out, int n, int m);
Compute a 1D ADAPTIVE AVERAGE POOL: reduce n uint8 inputs to exactly m int8 outputs.
The input is split into m equal, contiguous windows of size k = n/m (n % m == 0):
    out[i] = (int8_t)( (in[i*k] + in[i*k+1] + ... + in[i*k+k-1]) / k )
using an int32 accumulator; the division truncates and the result is cast (wrapped)
to int8. For this task n = 1114112, m = 68, so k = 16384.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. Because k = 16384,
each window is exactly one 16KB DMA tile. To go fast, DMA each window from DDR into VTCM and
run the reduction on the fast on-chip copy, double-buffering so the next window's DMA overlaps
the current window's reduce. VTCM is identity-mapped at 0xd8400000. uDMA: static/global Type-0
descriptor {next,ctrl=len,src,dst}, Q6_dmstart_A(&desc)/Q6_R_dmwait(). An unsigned vrmpy
(Q6_Vuw_vrmpyacc_VuwVubVub) against a splat of 0x01010101 reduces 4 unsigned bytes into one
u32 lane; horizontally sum the 32 lanes per window, divide by k, and cast to int8.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
