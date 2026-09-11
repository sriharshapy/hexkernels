# Task: int8 2-way interleave (zip)

Merge two planar `int8` streams `a` and `b` of length `n` into a single
interleaved output of length `2*n`:

```
out[2*i]   = a[i]
out[2*i+1] = b[i]     for i in [0, n)
```

This is the planar->interleaved layout transform (e.g. stereo L/R planes to an
interleaved buffer, or a 2-channel NCHW->NHWC). Pure data movement — no
arithmetic on the values.

`n = 393216` (the output is 768 KB, so the working set exceeds L2 and the task
is DDR-bandwidth bound). The two concurrent input streams make a naive
direct-DDR implementation latency-bound; staging both streams in VTCM with a
double-buffered uDMA (prefetch the next tile while interleaving the current one
on-chip) recovers the gap.

Implement:

```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
```

HVX `Q6_W_vshuff_VVR(vb, va, -1)` produces a vector pair whose low/high halves
are the byte-interleave of `va` and `vb`.
