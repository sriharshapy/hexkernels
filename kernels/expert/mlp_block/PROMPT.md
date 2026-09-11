Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t  *A,
                          const int8_t  *W1, const int32_t *b1,
                          const int8_t  *gelu_lut,
                          const int8_t  *W2, const int32_t *b2,
                          int8_t        *out,
                          int M, int K, int V, int D,
                          int32_t mult, int shift, int8_t zp);

Two-layer MLP block (all integer):
  Layer 1 -- GEMM + bias + sat8 + GELU LUT:
    For each token i in [0,M), hidden dim v in [0,V):
      acc   = sum_k A[i*K+k] * W1[k*V+v]   (int32 accumulator)
      biased= acc + b1[v]
      pre   = sat8(biased)                   (clamp to [-128,127])
      hid[i*V+v] = gelu_lut[(uint8_t)(pre + 128)]   -- DO NOT hardcode the LUT

  Layer 2 -- GEMM + bias + requant:
    For each token i in [0,M), output dim j in [0,D):
      acc2  = sum_v hid[i*V+v] * W2[v*D+j]  (int32 accumulator)
      biased2 = acc2 + b2[j]
      v64   = (int64_t)biased2 * mult
      half  = shift > 0 ? (1LL << (shift-1)) : 0
      r     = (v64 >= 0) ? (v64+half)>>shift : -(((-v64)+half)>>shift)
      r    += zp
      out[i*D+j] = sat8(r)

Dims: M=8, K=16, V=32, D=16. mult, shift, zp, gelu_lut are runtime params -- do NOT hardcode.
Prefer HVX vrmpy for K-reductions. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
