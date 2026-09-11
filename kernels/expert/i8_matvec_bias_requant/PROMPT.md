Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *A, const int8_t *x,
                          const int32_t *bias, int8_t *out,
                          int M, int K,
                          int32_t mult, int shift, int8_t zp);
Fused matrix-vector multiply + bias + requantize -> int8 (fully-connected layer):
  Step 1 -- dot product: acc[i] = sum_k A[i*K+k]*x[k]   (int32 accumulator)
  Step 2 -- bias:        biased = acc[i] + bias[i]        (int32; bias is per output row)
  Step 3 -- requantize to int8:
    v    = (int64_t)biased * mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
    r   += zp
    out[i] = saturate_to_int8(r)   // clamp to [-128, 127]
A is [M x K] int8 row-major, x is [K] int8, bias is [M] int32, out is [M] int8.
M=64, K=128. mult, shift, zp are runtime params -- do NOT hardcode them.
Prefer HVX vrmpy for the K-dot. Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
