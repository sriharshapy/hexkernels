Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n);
Compute out[i] = clamp((a[i]*3 + 1) >> 1, -128, 127) for i in [0, n): a fixed-point
scale-by-1.5 with round-to-nearest (the "+1" before the shift), saturating to int8.
The ">>1" is an ARITHMETIC shift (floor(t/2), matching plain C ">>" on a signed int
and Q6_Vh_vasr_VhR on HVX) -- do NOT drop the "+1" rounding term.

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound over a SINGLE
stream. To go fast, issue a ROLLING l2fetch prefetch of a[] a fixed distance ahead of
the read pointer, continuously through the whole loop (not just once at the start),
and unroll the compute loop so more loads are in flight per prefetch window. l2fetch
has no C intrinsic on this toolchain -- use inline asm: l2fetch(addr, desc) where desc
packs {stride:dir, width, height} (see existing l2fetch tasks for the exact bit layout).
n is not necessarily a multiple of 128; handle any remainder.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
