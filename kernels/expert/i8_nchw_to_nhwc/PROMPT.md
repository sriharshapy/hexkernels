# Task: NCHW -> NHWC layout transform (C=4)

Convert a single-batch `int8` tensor from channel-first (NCHW) to channel-last
(NHWC) layout, with `C = 4` channels:

```
in : 4 planes, each HW contiguous bytes in[c*HW + p]
out : HW pixels, each 4 contiguous bytes out[p*4 + c] = in[c*HW + p]
```

With `C = 4` this is a 4-way interleave of the four channel planes. Pure data
movement — no arithmetic.

`HW = 196608` (a multiple of 128), so `in`/`out` are 768 KB each and the working
set exceeds L2 (DDR-bandwidth bound). The four channel planes are four
concurrent input streams a stride `HW` apart, which makes a naive direct-DDR
gather latency-bound; staging the planes in VTCM with a double-buffered uDMA
(prefetch the next tile while interleaving the current one on-chip) recovers the
gap.

Implement:

```c
void candidate_kernel(const int8_t *in, int8_t *out, int C, int HW); /* C == 4 */
```

Four-way byte interleave via two levels of HVX `Q6_W_vshuff_VVR`: interleave
planes (0,1) and (2,3) at byte granularity, then interleave those two results at
halfword granularity.
