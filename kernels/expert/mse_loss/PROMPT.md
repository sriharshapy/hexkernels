# MSE Loss (n=512)

Implement:
```c
void candidate_kernel(const float *pred, const float *tgt, float *out, int n);
```

**Semantics:** `out[0] = (1.0f/n) * sum_i( (pred[i] - tgt[i])^2 )`

Mean Squared Error reduction over n=512 elements. The result is a scalar stored in `out[0]`.

Inputs `pred` and `tgt` are fp32 arrays with values in `[-2, 2]`.

**Tolerance:** fp32 reduction (HVX qfloat path is not IEEE bit-exact).
Accepted if `|got - ref| <= 1e-4 + 1e-3 * |ref|`.

Use HVX fp32 vectorized multiply-add for throughput; accumulate partial sums
across 32-element vectors (128B / 4B each), then reduce the vector to a scalar
and divide by n.
