# HVX Drill: Elementwise int16 Multiply, Low Half (n=577)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
```

**Semantics:** `out[i] = (int16_t)((int)a[i] * (int)b[i])` for `i` in `[0, n)` — the
**low 16 bits** of the exact 32-bit product, reinterpreted as signed two's-complement
`int16`. This is a plain truncating **integer** multiply, matching `Q6_Vh_vmpyi_VhVh`.

Do NOT confuse this with a Q15 fixed-point fractional multiply (which right-shifts
the product by 15 before narrowing) — that intrinsic family is different and produces
a different result. For example `300 * 300 = 90000`, which truncates to `24464`
(NOT `90000 >> 15 = 2`).

- `n=577` (not a multiple of the 64-lane HVX int16 vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
