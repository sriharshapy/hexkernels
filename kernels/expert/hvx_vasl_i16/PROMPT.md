# HVX Drill: Arithmetic Shift-Left, int16 (n=583, shift=4)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int16_t *a, int16_t *out, int n, int shift);
```

**Semantics:** `out[i] = (int16_t)((uint16_t)a[i] << shift)` for `i` in `[0, n)`.

Bits shifted out of bit 15 are simply lost — this is a WRAPPING shift, NOT
a saturating one. This matches the HVX intrinsic `Q6_Vh_vasl_VhR`. For
example `0x4000 << 4` truncates to `0x0000` (wraps to `0`), it does not
saturate to `32767`.

- `shift` is always called with the fixed value `4` in this harness.
- `n=583` (not a multiple of the 64-halfword HVX vector width) — handle
  the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
