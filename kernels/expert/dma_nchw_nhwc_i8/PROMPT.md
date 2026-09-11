Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int N, int C, int H, int W);
`in` is an N x C x H x W (NCHW) row-major int8 tensor. Convert it to NHWC layout in `out`:
    out[((n*H+h)*W+w)*C + c] = in[((n*C+c)*H+h)*W+w]
    for n in [0,N), c in [0,C), h in [0,H), w in [0,W)
N=1, C=64, H=100, W=200 (C*H*W = 1.22MB per image, exceeds L2, so this is
DDR-bandwidth-bound). H*W=20000 is not a multiple of 128 -- the last band is
a clipped tail.

For a FIXED image, this is exactly a C x (H*W) -> (H*W) x C matrix transpose:
channel c is a row of H*W contiguous bytes in `in`; spatial position (h,w) is
a row of C contiguous bytes in `out`. There is no documented hardware "2D
descriptor" bit layout on this toolchain, so build the conversion out of: for
each band of TC=128 consecutive spatial positions (flattened h*W+w index), a
CHAIN of C per-channel Type-0 uDMA descriptors {next, ctrl=band_width,
src = image + c*H*W + p0, dst = <contiguous VTCM slot + c*slot_stride>},
linked via `next` and issued with a SINGLE Q6_dmstart_A call (C is small and
fixed, so this one trigger moves the whole C x band_width sub-rectangle).
Once the band lands in VTCM, transpose it ON-CHIP into a second VTCM buffer
densely packed (row stride C) so the transposed band is contiguous both in
VTCM and in `out`, then DMA it out with ONE flat descriptor. Double-buffer
across spatial bands so the next band's input gather-chain overlaps this
band's on-chip transpose + output DMA. Repeat per image (N=1 in this test,
but the design must generalize to N>1). VTCM is identity-mapped at
0xd8400000. Every descriptor MUST be static/global (a stack descriptor
no-ops at -O2). H*W is not necessarily a multiple of TC=128; handle the
clipped final band.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
