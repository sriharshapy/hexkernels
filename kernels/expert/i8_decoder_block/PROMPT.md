# int8 transformer DECODER BLOCK (causal self-attention + LUT-FFN, 2 residuals, HVX)

Implement
`candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1, const int8_t *act_lut, const int8_t *W2, const int32_t *b2, const uint8_t *exp_lut, int8_t *out, int S, int D, int Dff)`
computing a full transformer **decoder** block over `S=32` tokens of model-dim
`D=64` with FFN hidden `Dff=128`. `X` is uint8 (0..3, used directly as
self-attention Q,K,V), `W1` int8 (-3..3), `W2` int8 (-2..2):

```
--- CAUSAL self-attention sub-layer (Q = K = V = X) ---
for i in [0,S):
 for j in [0,i]: (CAUSAL: j<=i ONLY)
 raw = sum_d X[i*D+d] * X[j*D+d]
 scaled[i][j] = clamp( raw >> ATTN_SCALE_SHIFT, -128,127)
 probs[i][0..i] = softmax_lut(scaled[i][0..i]) (uint8, over the causal window only)
 for d in [0,D):
 av = sum_{j=0}^{i} probs[i][j] * X[j*D+d]
 a[i][d] = clamp( av >> ATTN_OUT_SHIFT, -128,127)

--- Residual 1 ---
h[i][d] = clamp_u8( X[i*D+d] + a[i*D+d] ) (uint8, FFN activation)

--- FFN sub-layer (LUT activation, NOT ReLU) ---
acc1[i][j] = sum_k h[i*D+k] * W1[k*Dff+j]
idx = clamp( (acc1>>FFN_SH1) + b1[j], -128,127) + 128 (0..255)
H[i][j] = act_lut[idx] (int8, runtime LUT)
acc2[i][j] = sum_k H[i*Dff+k] * W2[k*D+j]
f[i][j] = sat_i8( (acc2>>FFN_SH2) + b2[j] )

--- Residual 2 ---
out[i][d] = sat_i8( h[i*D+d] + f[i*D+d] ) (int8 block output)
```
(`ATTN_SCALE_SHIFT=2, ATTN_OUT_SHIFT=8, FFN_SH1=8, FFN_SH2=7`.)

This is a decoder block, NOT an encoder block: attention is **causal** — query
row `i` may only look at keys `j<=i` (positions `j>i` never enter the max,
the exp-sum, or the weighted-V accumulation). The FFN nonlinearity is a
**runtime lookup table** (`act_lut`, 256 entries, GELU-shaped) instead of a
closed-form ReLU — do NOT hardcode or approximate it, index it.

`exp_lut` is the same runtime fixed-point-exp softmax LUT scheme as the
encoder block (index = `255 - clamp(rowmax - scaled, 0, 255)`), also a
runtime input — do NOT hardcode it either.

Pure HVX (no HMX/VTCM needed). The whole block is fixed-point integer, so it
is **BIT-EXACT** to the reference — integer add/multiply is associative, so a
correctly-vectorized reduction is exact as long as it sums exactly the causal
window (no extra out-of-window terms) and the full real reduction length
elsewhere (D for the up-proj, Dff for the down-proj).

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.
