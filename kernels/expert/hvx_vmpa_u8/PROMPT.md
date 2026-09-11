# HVX Drill: Dual Weighted-Sum Multiply-Add (n=780)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b,
                      const uint8_t *wa, const uint8_t *wb,
                      int16_t *out, int n);
```

**Semantics:** `out[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i])`
for `i` in `[0, n)`. All of `a`, `b`, `wa`, `wb` are `uint8_t`. Use
two's-complement **WRAP** on int16 overflow (the hardware MAC does not
saturate). This matches the HVX intrinsic `Q6_Wh_vmpa_WubWub` applied
per-lane across the `(a,wa)` and `(b,wb)` operand pairs — it is a plain
per-index two-term weighted sum of two channels, NOT a 4-tap group
reduction like `vdmpy`.

- `n=780` (not a multiple of the 128-byte HVX vector width) — handle the
  scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
