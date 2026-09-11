# HVX Drill: 256-Entry Byte Lookup Table (n=900)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, int8_t *out, int n, const int8_t *lut);
```

**Semantics:** `idx = (uint8_t)a[i]` (reinterpret the signed byte's bit
pattern as an unsigned index in `[0,255]`); `out[i] = lut[idx]` for `i`
in `[0, n)`. `lut` has exactly 256 entries, supplied as a runtime
pointer (do NOT hardcode any values — the table is a task input). This
matches the HVX gather intrinsic `Q6_Vb_vlut32_VbVbR`/`Q6_Vb_vlut32or_VbVbVbR`.

- `n=900` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
