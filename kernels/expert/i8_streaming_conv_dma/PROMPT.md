Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
 int n, int ntaps, int shift);
Compute a 5-tap 1D FIR as CORRELATION (taps are NOT reversed), then requantize to int8:
 acc = sum over j=0..4 of x[i+j]*taps[j] (int32 accumulator), for i in [0, n)
 half = shift > 0 ? (1 << (shift-1)) : 0
 r = (acc >= 0) ? (acc+half)>>shift : -(((-acc)+half)>>shift) // round half away from zero
 out[i] = saturate_to_int8(r)
x has n+4 samples (ntaps-1=4 halo samples); out has n int8 results. ntaps is fixed to 5.
taps and shift are runtime parameters -- do NOT hardcode them. n is a multiple of 128.

n is large and the task is DDR-bandwidth-bound. To go fast, tile by output blocks: DMA the
input window (block + ntaps-1 halo bytes) from DDR into VTCM, run the FIR+requant on the
on-chip copy writing int8 results into a VTCM output slot, then DMA the output slot back to
DDR -- double-buffered so the next input window's DMA overlaps the current block's compute.
VTCM is identity-mapped at 0xd8400000. uDMA: static/global Type-0 descriptor
{next,ctrl=len,src,dst}, `Q6_dmstart_A(&desc)`/`Q6_R_dmwait()` (a stack descriptor no-ops at -O2).
Vectorize with widening int8*int8->int16 multiply-accumulate (unaligned input loads via
valign + a tap broadcast per tap), then round+shift+saturate-pack to int8. Handle any tail.

Respond with a single complete C code block and CLOSE the fence with ```.
