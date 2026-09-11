# HVX Drill: Q15 Fractional Multiply-High, int16 (n=519)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n);
```

**Semantics:** treat `a[i]` and `b[i]` as Q15 fixed-point values (range
`[-1, 1)`); compute their product and take the Q15 "high half", rounding
to nearest (ties away from zero) and saturating:
```
P = (int32_t)a[i] * (int32_t)b[i]
bias = (P >= 0) ? 16384 : -16384
q = truncate_toward_zero((P + bias) / 32768)
out[i] = sat16(q)
```
This matches the HVX intrinsic `Q6_Vh_vmpy_VhVh_s1_rnd_sat`. Example:
`a=16384 (0.5), b=16384 (0.5)` gives `8192` (`0.25`); `a=b=32767`
(both just under `1.0`) gives `32766`, not `32767` (the product is
slightly less than `1.0`).

- `n=519` (not a multiple of the 64-halfword HVX vector width) — handle
  the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
