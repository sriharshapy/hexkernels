# int8 MLP-MIXER block (token-mixing MLP + channel-mixing MLP, 2 residuals, HVX)

Implement
`candidate_kernel(const uint8_t *X, const int8_t *Wt1, const int32_t *bt1, const int8_t *Wt2, const int32_t *bt2, const int8_t *Wc1, const int32_t *bc1, const int8_t *Wc2, const int32_t *bc2, int8_t *out, int S, int D, int St, int Dff)`
computing an MLP-Mixer block over `S=32` tokens of `D=64` channels, with
token-mix hidden `St=64` and channel-mix hidden `Dff=64`. `X` is uint8 (0..3).

```
--- Token-mixing MLP (mixes over the SEQUENCE axis; Wt1/Wt2 SHARED across all channels d) ---
for d in [0,D):
 for s' in [0,St):
 acc1[s'][d] = sum_{s=0}^{S-1} X[s*D+d] * Wt1[s'*S+s]
 TMh[s'][d] = relu_i8( (acc1[s'][d] >> TM_SH1) + bt1[s'] ) (uint8, 0..127)
 for s in [0,S):
 acc2[s][d] = sum_{s'=0}^{St-1} TMh[s'][d] * Wt2[s*St+s']
 TMo[s][d] = sat_i8( (acc2[s][d] >> TM_SH2) + bt2[s] )
Y[s][d] = sat_i8( X[s*D+d] + TMo[s][d] ) (residual 1)

--- Channel-mixing MLP (mixes over the FEATURE axis; Wc1/Wc2 SHARED across all tokens s) ---
for s in [0,S):
 for j in [0,Dff):
 acc1[s][j] = sum_{d=0}^{D-1} Y[s*D+d] * Wc1[d*Dff+j]
 CMh[s][j] = relu_i8( (acc1[s][j] >> CM_SH1) + bc1[j] ) (uint8, 0..127)
 for d in [0,D):
 acc2[s][d] = sum_{j=0}^{Dff-1} CMh[s][j] * Wc2[j*D+d]
 CMo[s][d] = sat_i8( (acc2[s][d] >> CM_SH2) + bc2[d] )
out[s][d] = sat_i8( Y[s*D+d] + CMo[s][d] ) (residual 2)
```
`relu_i8(p) = (p>0) ? clamp(p,0,127) : 0`.
(`TM_SH1=2, TM_SH2=7, CM_SH1=8, CM_SH2=7`.)

This is the defining MLP-Mixer structure: TWO transposed matmul stages over
the SAME activation tensor. The token-mixing stage reduces over the sequence
axis `s` (weights `Wt1`/`Wt2` are shared identically across every channel
`d` — you must apply the same 1-D token MLP independently per channel); the
channel-mixing stage reduces over the feature axis `d` (weights `Wc1`/`Wc2`
shared across every token `s`, exactly like a normal transformer FFN). Each
stage has its own residual add.

Pure HVX (no HMX/VTCM needed). The whole block is fixed-point integer, so it
is **BIT-EXACT** to the reference — integer add/multiply is associative, so a
correctly-vectorized reduction is exact regardless of vectorization order, as
long as it sums exactly the real reduction length (`S`/`St` for token-mixing,
`D`/`Dff` for channel-mixing) with no extra terms.

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.
