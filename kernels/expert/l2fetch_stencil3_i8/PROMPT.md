Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n);
Compute, for i in [0, n):
    out[i] = clamp(a[clamp(i-1,0,n-1)] + 2*a[i] + a[clamp(i+1,0,n-1)], -128, 127)
a 3-tap weighted stencil (weights 1,2,1) with EDGE-REPLICATED boundaries (the missing
neighbor at i=0 / i=n-1 is replaced by the nearest in-range element, NOT zero). Do NOT
drop the "2*a[i]" center weight (a plain 1,1,1 average is a different, wrong, formula).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound over a SINGLE
stream. To go fast, issue a ROLLING l2fetch prefetch of a[] a fixed distance ahead of
the read pointer, continuously through the whole loop. l2fetch has no C intrinsic on
this toolchain -- use inline asm: l2fetch(addr, desc). Use unaligned 128B vector loads
(a `long __attribute__((vector_size(128)))` cast, aka "vmemu") for the shifted -1/+1
taps since they are not 128B-aligned. n is not necessarily a multiple of 128; handle
any remainder and the two boundary elements.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
