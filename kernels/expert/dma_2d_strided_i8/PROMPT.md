Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int H, int W, int rowstride);
`a` is an H x rowstride buffer where each row is padded to `rowstride` bytes
(rowstride >= W). Compute the compact depad copy:
    out[r*W + c] = a[r*rowstride + c]     for r in [0, H), c in [0, W)
H = 136, W = 8192, rowstride = 8320 (128-byte padding per row); the padded working
set exceeds L2, so this is DDR-bandwidth-bound.

There is no documented hardware "2D descriptor" bit layout on this toolchain, so build
the row-strided transfer out of a CHAIN of per-row Type-0 uDMA descriptors: one
descriptor per row {next, ctrl=W, src = a + row*rowstride, dst = <contiguous VTCM
slot + row_in_group*W>}, linked via `next` and issued with a SINGLE Q6_dmstart_A call
for a group of rows, then Q6_R_dmwait(). Once a row-group lands contiguously in VTCM,
DMA that whole contiguous buffer out to `out` with one flat descriptor. Double-buffer
across row-groups so the next group's row-chain DMA overlaps the previous group's
DMA-out. VTCM is identity-mapped at 0xd8400000. Every descriptor MUST be static/global
(a stack descriptor no-ops at -O2). H is not necessarily a multiple of the row-group
size; handle any remaining rows.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
