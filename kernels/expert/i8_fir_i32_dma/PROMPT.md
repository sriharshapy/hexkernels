Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out, int n, int ntaps);
Compute a 1D FIR as CORRELATION (taps are NOT reversed):
    out[i] = sum over j=0..ntaps-1 of x[i+j]*taps[j],   for i in [0, n).
x has n+ntaps-1 samples; out has n int32 results. Use a 32-bit signed accumulator.
ntaps is fixed to 8. n is a multiple of 128.

n is large and the int32 output dominates memory traffic, so the task is DDR-bandwidth-bound.
To go fast, tile by output blocks: DMA the input window (block + ntaps-1 halo bytes) from DDR
into VTCM, run the FIR on the on-chip copy writing int32 results into a VTCM output slot, then
DMA the output slot back to DDR — double-buffered so the next input window's DMA overlaps the
current block's compute. VTCM is identity-mapped at 0xd8400000. uDMA: static/global Type-0
descriptor {next,ctrl=len,src,dst}, Q6_dmstart_A(&desc)/Q6_R_dmwait(). Vectorize the FIR with
widening int8*int8->int32 multiply-accumulate (unaligned input loads via valign + a tap
broadcast per tap). Handle any tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
