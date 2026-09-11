Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int w, int h);

Fused 2x2 NON-overlapping max pool (signed int8) then ReLU (window/stride/ReLU are FIXED
constants of this task). Output is (w/2) x (h/2), row-major:
    pool = max(in[2oy][2ox], in[2oy][2ox+1], in[2oy+1][2ox], in[2oy+1][2ox+1])
    out[oy*(w/2)+ox] = pool > 0 ? pool : 0     (ReLU; output is always in [0,127])
w and h are even. Inputs are signed (can be negative); after ReLU the output is >= 0.

The image is large (working set exceeds L2), so this is DDR-bandwidth-bound. To go fast,
DMA row blocks of in[] from DDR into VTCM and pool+relu the fast on-chip copies,
double-buffering so the next block's DMA overlaps the current block's compute; DMA results
back to DDR. Non-overlapping windows use disjoint input rows, so no halo is needed. VTCM
is identity-mapped at 0xd8400000. uDMA: build a static/global Type-0 descriptor
{next,ctrl=len,src,dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait(). The descriptor MUST be
static/global (a stack descriptor no-ops at -O2). Use signed Q6_Vb_vmax_VbVb for max and
column deinterleave via Q6_W_vdeal_VVR; ReLU is a vmax against zero.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
