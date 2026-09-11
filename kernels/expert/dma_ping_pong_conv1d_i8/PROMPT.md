Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *w, int32_t bias, int shift,
                          int8_t *out, int n);
Compute, for i in [0,n), a runtime 3-tap FIR convolution with EDGE-REPLICATED
boundaries (NOT zero-padding) plus a bias+shift requantize:
    left   = a[clamp(i-1, 0, n-1)]
    center = a[i]
    right  = a[clamp(i+1, 0, n-1)]
    t      = w[0]*left + w[1]*center + w[2]*right + bias      (int32)
    sm     = t >> 31 (arithmetic)          -- 0 if t>=0, -1 if t<0
    abs    = (t ^ sm) - sm                 -- |t|
    half   = shift > 0 ? (1 << (shift-1)) : 0
    sh     = (abs + half) >> shift (arithmetic)
    r      = (sh ^ sm) - sm
    out[i] = clamp(r, -128, 127)
w (3 int8 taps), bias, shift are runtime params (harness fixes w={2,5,3},
bias=10, shift=4). There is no separate multiplier here (mult=1 implicit --
the taps already scale the sum), so skip any vmpyie-style multiply and use
abs directly.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To
go fast, DMA a[] in CH=16384-byte tiles from DDR into VTCM, but give each
tile's on-chip buffer a 1-byte HALO on each side (layout: [halo_left(1B)]
[core(CH B)][halo_right(1B)], CH+2 bytes total) so the conv can be computed
correctly across tile boundaries without re-touching DDR. For an interior
tile (not the first) DMA the halo'd CH+2-byte region as ONE contiguous
descriptor: src = a + tile_base - 1, len = CH+2. The FIRST tile has no
a[-1]: DMA only CH+1 bytes starting at a[0] into buffer offset 1, then
replicate buf[0] = buf[1] after the DMA completes (one scalar byte copy).
Symmetrically, if a tile ever butts exactly against n with no right
neighbor available, DMA only CH+1 bytes and replicate buf[CH+1] = buf[CH]
after the DMA completes. Double-buffer this across tiles (prefetch tile
c+1's halo'd region while computing tile c), the same way the DMA
double-buffer patterns elsewhere in this benchmark do. Only stream FULL
tiles this way (nfull = n/CH); handle the scalar remainder with the plain
edge-replicated scalar formula directly against a[] (tiny, DDR-direct is
fine).

For an output block of 128 at local tile offset v*128, the taps come from
UNALIGNED loads into the halo'd buffer: left tap = buf[v*128 .. v*128+127],
center tap = buf[v*128+1 .. +128], right tap = buf[v*128+2 .. +128] (buf[k]
holds a[tile_base+k-1]). Widen int8->int16 (Q6_Wh_vunpack_Vb), multiply each
tap's int16 values by its splatted int8 weight (Q6_Vh_vmpyi_VhVh with
Q6_Vh_vsplat_R), accumulate in int16 (safe: max |sum| is tiny), widen the
int16 sum to int32 (Q6_Ww_vunpack_Vh), add the splatted int32 bias, requant
per the formula above, and saturate-pack back to int8
(Q6_Vh_vpack_VwVw_sat then Q6_Vb_vpack_VhVh_sat). VTCM is identity-mapped at
0xd8400000. uDMA: a static/global Type-0 descriptor {next, ctrl=len, src,
dst}, then Q6_dmstart_A(&desc)/Q6_R_dmwait().

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
