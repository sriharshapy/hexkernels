Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int16_t *out, int n,
                          int8_t zp, int32_t scale, int shift);
For each i in [0,n): v = ((int32_t)a[i] - (int32_t)zp) * scale;
out[i] = (int16_t)clamp(v >> shift, -32768, 32767). The shift is an ARITHMETIC
right shift (rounds toward -inf, no round-half-away). zp/scale/shift are runtime
params -- do NOT hardcode them. Inputs are bounded so (a-zp)*scale fits int16.

n is large (int8 read + int16 write => DDR-bandwidth-bound). To go fast you should hide
DDR latency: stage tiles of a[] into VTCM with the uDMA engine and compute on the on-chip
copies, double-buffering so the next tile's DMA overlaps the current tile's compute; DMA
results back to DDR. VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global
Type-0 descriptor {next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait(). The
descriptor MUST be static/global (a stack descriptor no-ops at -O2). Widen int8 to int16
with Q6_Wh_vsxt_Vb and work in halfword lanes (two halfword sub-vectors per input
byte-vector). n is a multiple of 128; handle any sub-tile remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
