# HMX int8 requant-store epilogue (int32 accumulators -> saturated int8)

Implement `candidate_kernel(const int32_t *acc, int8_t *out, int n)` for `n == 1024`.

`acc` holds the int32 accumulators an HMX matmul produced. Apply the fixed
`0x40`-config requant and store as signed int8 with saturation:

```
r      = (acc[i]*17 + 8) >> 4        (arithmetic shift right; the HMX 0x40 field)
out[i] = clamp(r, -128, 127)          (signed int8 saturating store)
```

- No matmul here -- this is the quantize+store tail that turns int32 accumulators
  into an int8 tensor.
- The input range makes `r` exceed the int8 range for most elements, so the
  saturation must be correct (a plain truncating cast fails). Boundary
  accumulators (r == 127, 128, -128, -129) are seeded to pin the clamp edges.
- `acc` is 128-byte aligned; you may process it with HVX word vectors.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
code block.
