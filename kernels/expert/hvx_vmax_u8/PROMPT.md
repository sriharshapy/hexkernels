# HVX Drill: Elementwise Unsigned Max, uint8 (n=1013)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = (a[i] > b[i]) ? a[i] : b[i]` for `i` in `[0, n)`,
using **unsigned** comparison. This matches the HVX intrinsic
`Q6_Vub_vmax_VubVub` (NOT the signed `Q6_Vb_vmax_VbVb`). For example
`max(128, 127) = 128` — a signed compare would (wrongly) say `127`, since
`128` as a signed byte is `-128`.

- `n=1013` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
