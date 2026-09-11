Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *A, const int8_t *x,
 const int32_t *bias, int8_t *out,
 int M, int K,
 int32_t mult, int shift, int8_t zp);
Fused linear (FC) + ReLU + requantize -> int8:
 Step 1 -- dot product: acc[i] = sum_k A[i*K+k]*x[k] (int32 accumulator)
 Step 2 -- bias: biased = acc[i] + bias[i] (int32; bias is per output neuron)
 Step 3 -- relu: after_relu = max(biased, 0)
 Step 4 -- requantize to int8:
 v = (int64_t)after_relu * mult
 half = shift > 0 ? (1LL << (shift-1)) : 0
 r = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 r += zp
 out[i] = saturate_to_int8(r) // clamp to [-128, 127]
A is [M x K] int8 row-major (weights), x is [K] int8 (input), bias is [M] int32, out is [M] int8.
M=64, K=128. mult, shift, zp are runtime params -- do NOT hardcode them.
Respond with a single complete C code block and CLOSE the fence with ```.
