# HVX Drill: Elementwise Rounding uint8 Average (n=900)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const uint8_t *a, const uint8_t *b, uint8_t *out, int n);
```

**Semantics:** `out[i] = (uint8_t)(((int)a[i] + (int)b[i] + 1) >> 1)` for `i` in `[0, n)`
— the **round-half-up** average of `a[i]` and `b[i]`.

This matches the HVX intrinsic `Q6_Vub_vavg_VubVub_rnd`. Note the `+1` before the
shift: it is NOT the same as the floor/truncating `Q6_Vub_vavg_VubVub`
(`(a+b)>>1`, no `+1`) — the two differ whenever `a[i]+b[i]` is odd.

- `n=900` (not a multiple of the 128-byte HVX vector width) — handle the scalar tail.
- Arrays are 128-byte aligned (`HVX_ALIGN`).

Include `<hexagon_types.h>` and `<hexagon_protos.h>`. Do NOT write main().
Respond with a single complete C code block.
