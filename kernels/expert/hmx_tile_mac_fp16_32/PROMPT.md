# FP16 32x32 fused tile MAC on the HMX matrix engine

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out, int n)`
for `n == 32`. The output buffer arrives **pre-loaded with an accumulator tile
`C0`**; you must compute the multiply-accumulate

```
out[i*n+j] = C0[i*n+j] + sum_k A[i*n+k] * B[k*n+j]
```

i.e. ADD the A*B product into the existing output (do NOT overwrite it).

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE` for crouton packing.
- HVX/HMX float arithmetic is non-IEEE (qf16), so output is compared to a scalar
  float-accumulate reference with an fp16 tolerance (NOT bit-exact).
- The HMX FP16 path uses a 32x32 "crouton" tile: offset `(r/2)*64 + c*2 + (r&1)`
  for activation, weight, and output (`hvx_crouton_off`). Pipeline:
  `Q6_mxclracc_hf` -> `Q6_activation_hf_mxmem_RR` + `Q6_weight_hf_mxmem_RR`
  (limit 2047) -> `Q6_mxmem_AR_after_hf` -> isync, then add the product into C0
  during the un-crouton unpack.
- Stage the crouton pack/unpack through cacheable buffers and move to/from VTCM
  in bulk 128B vector copies; the HMX MAC must beat the HVX qf16 baseline by >=1.2x.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>`,
`<hexagon_protos.h>` and `<hmx_hexagon_protos.h>`. Do NOT write `main()`. Respond
with a single complete C code block.


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
