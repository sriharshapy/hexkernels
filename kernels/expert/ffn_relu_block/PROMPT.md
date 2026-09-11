Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *A,
 const int8_t *W1, const int32_t *b1,
 const int8_t *W2, const int32_t *b2,
 int8_t *out,
 int M, int K, int V, int D,
 int32_t mult, int shift, int8_t zp);

Two-layer FFN block with ReLU (all integer):
 Layer 1 -- GEMM + bias + ReLU + sat8:
 For each token i in [0,M), hidden dim v in [0,V):
 acc = sum_k A[i*K+k] * W1[k*V+v] (int32)
 biased = acc + b1[v]
 relu = max(0, biased)
 hid[i*V+v] = sat8(relu) (clamp to [-128,127])

 Layer 2 -- GEMM + bias + requant:
 For each token i in [0,M), output dim j in [0,D):
 acc2 = sum_v hid[i*V+v] * W2[v*D+j] (int32)
 biased2= acc2 + b2[j]
 v64 = (int64_t)biased2 * mult
 half = shift > 0 ? (1LL << (shift-1)) : 0
 r = (v64 >= 0) ? (v64+half)>>shift : -(((-v64)+half)>>shift)
 r += zp
 out[i*D+j] = sat8(r)

Dims: M=8, K=16, V=32, D=16. mult, shift, zp are runtime params -- do NOT hardcode.
Respond with a single complete C code block and CLOSE the fence with ```.
