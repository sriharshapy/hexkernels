# int8 MULTI-STEP GRU block (fused update/reset gate matmul, T timesteps, HVX)

Implement
`candidate_kernel(const int8_t *Wzr, const int8_t *Wn, const int32_t *bzr, const int32_t *bn, const int8_t *x, const int8_t *h0, int8_t *h_out, int T, int H, int I, int32_t mult, int shift, int8_t zp, const int8_t *sig_lut, const int8_t *tanh_lut)`
computing `T=3` steps of an integer GRU cell with hidden size `H=16` and
per-step input size `I=16` (`C=I+H=32`), batch=1. Unlike a single-step GRU
cell, this is a **block**: it recurs over the input sequence and a **fused**
update/reset gate matmul (`Wzr` stacks `Wz` then `Wr` row-wise: `[2H x C]`,
ONE weight tensor computes BOTH gates' pre-activations in one loop, since
they share the exact same input `concat[x_t | h_prev]`).

```
h_prev = h0
for t in [0,T):
 concat_xh = [x[t*I .. t*I+I-1], h_prev[0..H-1]] (C = I+H elements)

 for j in [0,2H): (FUSED z/r gate matmul)
 acc[j] = bzr[j] + dot(Wzr[j,:], concat_xh)
 q[j] = requant(acc[j], mult, shift, zp)
 z[j] = sig_lut[(uint8_t)q[j]] for j in [0,H)
 r[j] = sig_lut[(uint8_t)q[H+j]] for j in [0,H)

 rh[j] = clamp( ((r[j]+128)*h_prev[j] + 64) >> 7, -128,127 ) for j in [0,H)

 concat_xrh = [x[t*I..], rh[0..H-1]]
 accn[j] = bn[j] + dot(Wn[j,:], concat_xrh)
 n[j] = tanh_lut[(uint8_t)requant(accn[j], mult, shift, zp)]

 h_t[j] = clamp( ((128-z[j])*h_prev[j] + (z[j]+128)*n[j] + 128) >> 8, -128,127 )
 h_out[t*H .. t*H+H-1] = h_t[:]
 h_prev = h_t (propagate to next step)
```

`requant(acc,mult,shift,zp)`: `v=(int64_t)acc*mult; half=shift>0?1<<(shift-1):0;
r = v>=0 ? (v+half)>>shift : -((-v+half)>>shift); r+=zp; clamp(r,-128,127)`.

`Wzr`/`Wn` columns are `[x-part | h-part]`: `Wzr[j,0:I]`/`Wn[j,0:I]` multiply
`x_t`; `Wzr[j,I:I+H]` multiplies `h_prev` (update/reset gates); `Wn[j,I:I+H]`
multiplies `rh`, NOT `h_prev` (the candidate's recurrent input is the
RESET-gated hidden state). `mult`, `shift`, `zp`, `sig_lut`, `tanh_lut` are
ALL runtime inputs (swept across multiple configurations) — do NOT hardcode
any of them.

You MUST propagate `h_t` as `h_prev` into the NEXT timestep (a common bug is
to keep reusing `h0` for every step). Pure HVX (no HMX). The whole block is
plain fixed-point integer arithmetic, so it is **BIT-EXACT** to the reference.

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.
