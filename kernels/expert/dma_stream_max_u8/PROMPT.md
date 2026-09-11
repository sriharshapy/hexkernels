Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, int n, uint8_t *out);
Compute out[0] = max(a[0..n)) : the UNSIGNED uint8 maximum over all n elements.

n is large (working set exceeds L2), so this is a single DDR-bandwidth-bound reduction
pass. To go fast, DMA double-buffer a[] through VTCM in tiles: prefetch the next tile's
data via uDMA while reducing the current (already-resident) tile's max on-chip with
Q6_Vub_vmax_VubVub, accumulating a running per-lane max vector across tiles. At the
end, reduce the 128 lanes of the running-max vector down to a single scalar max (and
fold in the scalar tail). VTCM is identity-mapped at 0xd8400000. uDMA: build a
static/global Type-0 descriptor {next, ctrl=len, src, dst}, then
Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global. n is not
necessarily a multiple of the tile size or of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
