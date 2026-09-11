# int8 matrix-vector product (GEMV, N=1) on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *x, int32_t *out,
int n)` for `n == 32`:

```
acc[i] = sum_k A[i*n+k] * x[k]
out[i] = sign_extend_12bit((acc*17 + 8) >> 4)
```

`A` is `[M x K]` (uint8 0..7), `x` is `[K]` (int8 -3..3), `out` is `[M]` int32.
The `(acc*17+8)>>4 & 0xFFF` requant is the HMX matrix engine's native output at
the `0x40` bias-config.

- The matrix engine is a 32x32 tile. Map the GEMV by placing `x` in weight
  column 0 (`aWgt[hvx_hmx_i8_wgt_off(k,0)] = x[k]`) with all other weight columns
  zero, and read the result from output column 0 (`hvx_hmx_i8_out_off(i,0)`).
- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE`. int8 activation in the HIGH byte of a crouton slot
  (`hvx_hmx_i8_act_off`).
- Sequence: `mxclracc` -> `{ activation.ub=mxmem(...); weight.b=mxmem(...) }` ->
  `bias=mxmem(0x40-fill)` -> `mxmem(...):after.uh=acc:2x1` -> isync, then
  sign-extend the 12-bit field.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
code block.
