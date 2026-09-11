# HVX Drill: Sign-Extending Unpack int8 -> int16 (n=1000)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int16_t *out, int n);
```

**Semantics:** `out[i] = (int16_t)(int8_t)a[i]` for `i` in `[0, n)` — a plain
**sign-extending** widen. This matches the HVX intrinsic `Q6_Wh_vunpack_Vb`
(NOT the zero-extending `Q6_Wuh_vunpack_Vub`). For example `a[i] = -128`
must give `out[i] = -128` (not `128`).

- `n=1000` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
