# HVX Drill: Logical Shift-Right, uint16 (n=619, shift=5)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint16_t *a, uint16_t *out, int n, int shift);
```

**Semantics:** `out[i] = (uint16_t)(a[i] >> shift)` for `i` in `[0, n)` — a
plain **logical** (zero-fill) shift-right. This matches the HVX intrinsic
`Q6_Vuh_vlsr_VuhR`. Since the dtype is unsigned, there is no sign
extension: `0x8000 >> 5` must give `0x0400` (1024), not a value with
high bits set.

- `shift` is always called with the fixed value `5` in this harness.
- `n=619` (not a multiple of the 64-halfword HVX vector width) — handle
  the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
