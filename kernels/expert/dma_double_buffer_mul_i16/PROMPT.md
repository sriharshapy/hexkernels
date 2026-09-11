Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
Compute out[i] = (int16_t)(a[i] * b[i]) for i in [0, n): a low-16-bit truncating
multiply (wraparound on overflow, matching Q6_Vh_vmpyi_VhVh; NOT a widening or
saturating multiply).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
ping-pong two VTCM buffers for a, b, and out over the N tiles: prefetch the next
tile's a[]/b[] via uDMA while computing the current tile's multiply, and DMA the
result back out, so DDR latency is hidden behind compute. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is not necessarily a multiple of the tile size or
of 64 (the int16 vector width); handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
