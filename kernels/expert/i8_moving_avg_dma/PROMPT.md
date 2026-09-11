Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, int8_t *out, int n, int W);
Compute a 1D box moving average (VALID window, no padding), with window W fixed to 8:
    out[i] = (int8_t)( sum_{j=0}^{7} (int32)x[i+j] / 8 ),   for i in [0, n).
Division truncates TOWARD ZERO (standard C integer division, NOT an arithmetic shift /
floor). x has n+7 samples; out has n int8 results. n is a multiple of 128.

n is large and both the input and the output are streamed, so the task is DDR-bandwidth-bound.
To go fast, tile by output blocks: DMA the input window (block + 7 halo bytes) from DDR into
VTCM, run the moving average on the on-chip copy writing int8 results into a VTCM output slot,
then DMA the output slot back to DDR — double-buffered so the next input window's DMA overlaps
the current block's compute. VTCM is identity-mapped at 0xd8400000. uDMA: static/global Type-0
descriptor {next,ctrl=len,src,dst}, Q6_dmstart_A(&desc)/Q6_R_dmwait(). Keep compute light: form
the window sum with shifted int16 adds (valign, no multiply), divide by 8 with a
truncate-toward-zero correction, and pack the int16 results back to int8. Handle any tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
