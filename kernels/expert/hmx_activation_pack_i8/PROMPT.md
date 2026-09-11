# HMX int8 activation crouton pack (32x32 tile)

Implement `candidate_kernel(const uint8_t *A, uint8_t *out, int n)` for `n == 32`.

`A` is a row-major 32x32 uint8 activation tile. Produce `out`, the 2048-byte
crouton-packed activation buffer that the HMX `activation.ub=mxmem(...)` load
consumes. The int8 activation value lives in the HIGH byte of each fp16-crouton
16-bit slot, and every two logical rows are transposed:

```
out[ 2*((i/2)*64 + k*2 + (i&1)) + 1 ] = A[i*n + k]     for i,k in [0,32)
all other bytes of out                = 0               (the 1024 low/even bytes)
```

- No HMX instruction is required: this is the pure index remap performed before
  the mxmem load. The helper `hvx_hmx_i8_act_off(i,k)` in `harness_common.h`
  returns the destination byte offset if you want it.
- `out` is exactly 2048 bytes; ZERO it first (the even/low bytes must be 0).
- Bit-exact: every one of the 2048 output bytes is compared.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
code block.
