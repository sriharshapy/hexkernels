Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *op, int8_t *out, int n);
Compute out[i] = clamp(a[i] + op[i % 128], -128, 127) for i in [0, n): SATURATING int8
add of a large stream a[] with a small 128-byte (exactly one HVX vector) operand op[],
broadcast cyclically. op[] is EXACTLY 128 bytes -- load it ONCE into a vector register
and reuse it unchanged for every 128-byte chunk of a[] (do not reload/re-DMA op per
chunk, and no modulo arithmetic is needed inside the loop since 128 % 128 == 0).

n is large (working set exceeds L2), so a[] is DDR-bandwidth-bound. To go fast,
DMA double-buffer a[] through VTCM in tiles: prefetch the next tile via uDMA while
computing the current tile's saturating add against the once-loaded op vector, and
store results back out. VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next, ctrl=len, src, dst}, then
Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global. n is not
necessarily a multiple of the tile size or of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
