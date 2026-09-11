Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);

Compute a 2x2 NON-overlapping AVERAGE pool with stride 2, integer truncation, no rounding
(window/stride/truncate are FIXED constants of this task). Output is (w/2) x (h/2),
row-major:
    out[oy*(w/2)+ox] = (in[2oy][2ox] + in[2oy][2ox+1] + in[2oy+1][2ox] + in[2oy+1][2ox+1]) / 4.
w and h are even.

The image is large (working set exceeds L2), so this is DDR-bandwidth-bound. To go fast,
DMA row blocks of in[] from DDR into VTCM and pool the fast on-chip copies,
double-buffering so the next block's DMA overlaps the current block's compute; DMA the
pooled rows back to DDR. Non-overlapping windows use disjoint input rows, so no halo is
needed. VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global Type-0
descriptor {next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait(). The descriptor
MUST be static/global (a stack descriptor no-ops at -O2). Deinterleave columns with
Q6_W_vdeal_VVR, widening-add the four pixels in i16 (Q6_Wh_vadd_VubVub), shift right 2,
saturating-pack to u8 (Q6_Vub_vsat_VhVh).

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
