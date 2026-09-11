# HVX Drill: Elementwise int16 Subtract (n=651)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
```

**Semantics:** `out[i] = (int16_t)((int)a[i] - (int)b[i])` for `i` in `[0, n)`.

Use two's-complement **wraparound** on overflow — this matches the HVX intrinsic
`Q6_Vh_vsub_VhVh` (NOT the saturating `Q6_Vh_vsub_VhVh_sat`). For example
`-32768 - 1` wraps to `32767`.

- `n=651` (not a multiple of the 64-lane HVX int16 vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
