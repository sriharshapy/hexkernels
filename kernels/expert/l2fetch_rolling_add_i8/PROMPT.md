Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
Compute out[i] = a[i] + b[i] for i in [0, n) with two's-complement int8 wraparound
(matching Q6_Vb_vadd_VbVb; NOT saturating).

n is large (working set exceeds L2, DDR-bandwidth-bound). To go fast, roll a
continuous L2 prefetch ahead of BOTH input streams: every 2KB of progress, issue an
`l2fetch` for the next 2KB block of a[] and of b[] (a fixed distance ahead of the
current position), for the whole length of the loop -- not just once at the start.
Combine this with 4x loop unrolling for more in-flight loads. l2fetch has no C
intrinsic on this toolchain; use inline asm:
    uint64_t desc = ((uint64_t)16 << 32) | ((uint64_t)128 << 16) | (uint64_t)128;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(ptr), "r"(desc) : "memory");
n is a multiple of 128; handle any tail if you choose a different unroll factor.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
