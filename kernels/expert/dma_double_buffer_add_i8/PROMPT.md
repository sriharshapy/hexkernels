Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
Compute out[i] = clamp(a[i] + b[i], -128, 127) for i in [0, n): SATURATING int8 add
(matching Q6_Vb_vadd_VbVb_sat; NOT two's-complement wraparound).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
ping-pong two VTCM buffers (for a, b, and out) over the N tiles: prefetch the next
tile's a[]/b[] via uDMA while computing the current tile's saturating add, and DMA the
result back out, so DDR latency is hidden behind compute. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is not necessarily a multiple of the tile size or
of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
