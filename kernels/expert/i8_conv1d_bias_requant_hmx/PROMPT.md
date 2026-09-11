# int8 dense 1D conv (VALID) + bias + int8 requant on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
int8_t *out, int n)` computing a DENSE cross-channel 1D convolution with a fused
per-output-channel bias + int8 requant epilogue, for the fixed shape `C_in=16`,
`C_out=64`, `Kw=5`, input length `L_in=68`, output length `OL=64`:

```
acc[co][ol] = sum_{ci,kw} in[ci][ol+kw] * W[co][ci][kw]      (in uint8, W int8)
r   = sign_extend_12( (acc*17 + 8) >> 4 )                    (native HMX 0x40 requant)
out[co*OL + ol] = clamp(r + bias[co], -128, 127)            (fused bias + int8 requant)
```

- Lower to a matmul via **im2col**: rows = output positions (M=OL), reduction = the
  `(ci,kw)` window (K = C_in*Kw = 80), cols = output channels (N=C_out). K=80 is
  zero-padded to 3 crouton K-tiles.
- **Reuse the im2col activation:** pack an M-tile's activation into VTCM once and reuse it
  across all output-channel (N) tiles — this is what lets the HMX matmul throughput
  amortize the im2col cost (repacking per N-tile loses).
- Keep the pack division-free (`koff[k] = ci*L_in + kw`, `poff[p] = ol`; weight value =
  `W[co*K + kg]`); the harness has enabled HMX; use VTCM scratch at `HVX_VTCM_BASE`.
- Sign-extend the 12-bit 0x40 requant field, add the per-channel int32 `bias[co]`, and
  saturate to int8 — fuse into the HVX unpack. Stage crouton pack/unpack via cacheable
  buffers + bulk 128B vector copies.
- A competent HVX im2col+`vrmpy` conv1d with the same epilogue is the baseline; the HMX
  expert must beat it by >=1.2x.


## Available helper (HMX matrix engine)

You MAY `#include "hmx_helpers.h"` and call these sim-verified helpers instead of hand-writing
the crouton `mxmem` sequence. Each owns the crouton pack + `mxclracc`/`mxmem` matmul + VTCM
staging for `M,N,K` all multiples of 32; you do the epilogue (bias/relu/residual/requant/cast)
yourself in HVX or scalar.

int8 (A uint8 row-major MxK, B int8 row-major KxN):
```
void hmx_tile_matmul_i8(const uint8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
    /* C = sx12((sum_k A*B)*17+8 >> 4): the HMX 0x40-config 12-bit requant field, sign-extended */
void hmx_tile_matmul_i8_field(const uint8_t *A, const int8_t *B, uint16_t *field, int M, int N, int K);
    /* the raw uint16 crouton field, before sign-extension */
```
fp16 (tolerance comparison, not bit-exact):
```
void hmx_tile_matmul_fp16(const __fp16 *A, const __fp16 *B, float *C, int M, int N, int K);
void hmx_tile_matmul_fp16_field(const __fp16 *A, const __fp16 *B, __fp16 *field, int M, int N, int K);
```
Using the helper is optional — a correct kernel by any means is accepted.
