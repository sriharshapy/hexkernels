Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int32_t *a, const int32_t *b, int8_t *out, int n,
 int32_t mult, int shift, int8_t zp);
Residual-add requantize (skip-connection path): for each i in [0, n) compute:
 Step 1 -- widened add: sum = (int64_t)a[i] + (int64_t)b[i] // must NOT overflow
 Step 2 -- requantize to int8 (round-half-away-from-zero):
 v = sum * (int64_t)mult
 half = shift > 0 ? (1LL << (shift-1)) : 0
 r = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
 r += zp
 out[i] = saturate_to_int8(r) // clamp to [-128, 127]
n=1024 -- handle the tail. mult, shift, zp are runtime params (do NOT hardcode).
Respond with a single complete C code block and CLOSE the fence with ```.
