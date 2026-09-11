Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int n, int64_t *out);
Compute out[0] = sum over i in [0,n) of a[i], accumulated exactly (matches an
int64 running sum). Inputs are bounded to |a[i]| <= 1000.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go
fast, DMA tiles of a[] from DDR into VTCM and reduce the on-chip copies with a
32-lane int32 vector accumulator (Q6_Vw_vadd_VwVw), double-buffering so the
next tile's DMA overlaps the current tile's accumulate. VTCM is
identity-mapped at 0xd8400000. uDMA: a static/global Type-0 descriptor
{next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait(). The
descriptor MUST be static/global (a stack descriptor no-ops at -O2).

Because inputs are bounded to +-1000, each of the 32 int32 vector-accumulator
lanes stays far below overflow across the whole array -- only widen to int64
at the very END, when you cross-lane-reduce the 32 int32 lanes to a scalar
(and add any scalar tail read directly from DDR). n is not necessarily a
multiple of the DMA tile size; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
