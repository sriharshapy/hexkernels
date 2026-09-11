# HMX int8 weight crouton pack (32x32 tile, 4-deep)

Implement `candidate_kernel(const int8_t *B, int8_t *out, int n)` for `n == 32`.

`B` is a row-major 32x32 int8 weight tile with contraction index `k` as the row
and output column `j` as the column. Produce `out`, the 1024-byte 4-deep packed
weight buffer that the HMX `weight.b=mxmem(...)` load consumes:

```
out[ (k/4)*128 + j*4 + (k%4) ] = B[k*n + j]     for k,j in [0,32)
```

i.e. 4 consecutive contraction elements are packed into 4 consecutive bytes,
128 bytes per contraction-group x 8 groups = the dense 1024-byte tile. Every
output byte is written (no padding, unlike the activation tile).

- No HMX instruction is required: this is the pure index remap performed before
  the mxmem weight load. Helper `hvx_hmx_i8_wgt_off(k,j)` returns the byte offset.
- `out` is exactly 1024 bytes; all are compared bit-exact (signed int8).

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
code block.
