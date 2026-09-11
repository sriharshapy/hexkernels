# HVX Drill: Elementwise Unsigned Min, uint8 (n=947)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = (a[i] < b[i]) ? a[i] : b[i]` for `i` in `[0, n)`,
using **unsigned** comparison. This matches the HVX intrinsic
`Q6_Vub_vmin_VubVub` (NOT the signed `Q6_Vb_vmin_VbVb`). For example
`min(128, 127) = 127` — a signed compare would (wrongly) say `128`, since
`128` as a signed byte is `-128`, the smallest possible signed value.

- `n=947` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
