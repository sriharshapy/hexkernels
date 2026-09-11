Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *a, uint8_t *out, int n);
Compute out[i] = min(a[i] + 50, 255) (uint8 saturating add of the constant 50) for
i in [0, n).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast,
stage tiles of a[] into VTCM via uDMA, compute with HVX, and DMA results back out,
double-buffering across tiles. For this task, synchronize each DMA transfer by
*polling* rather than blocking: issue the transfer with Q6_dmstart_A(&desc), then spin
in a loop reading Q6_R_dmpoll() until it reports the transfer is complete, instead of
calling the blocking Q6_R_dmwait(). VTCM is identity-mapped at 0xd8400000. uDMA: build
a static/global Type-0 descriptor {next, ctrl=len, src, dst}. The descriptor MUST be
static/global (a stack descriptor no-ops at -O2). n is not necessarily a multiple of
the tile size or of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
