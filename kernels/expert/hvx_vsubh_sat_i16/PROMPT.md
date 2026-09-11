# HVX Drill: Saturating int16 Subtract (n=735)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
```

**Semantics:** `out[i] = sat16((int)a[i] - (int)b[i])` for `i` in
`[0, n)`, where `sat16` clamps to `[-32768, 32767]`. Use **saturating**
subtract — this matches the HVX intrinsic `Q6_Vh_vsub_VhVh_sat` (NOT the
wrapping `Q6_Vh_vsub_VhVh`). For example `30000 - (-10000)` must saturate
to `32767`, not wrap to a negative value.

- `n=735` (not a multiple of the 64-halfword HVX vector width) — handle
  the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
