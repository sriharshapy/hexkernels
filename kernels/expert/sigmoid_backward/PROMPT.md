# Sigmoid Backward Pass (n=512)

Implement:
```c
void candidate_kernel(const float *y, const float *dy, float *dx, int n);
```

**Semantics:** `dx[i] = dy[i] * y[i] * (1.0f - y[i])`

Sigmoid backward (gradient) pass: `y` is the sigmoid FORWARD OUTPUT in (0, 1),
`dy` is the upstream gradient, `dx` is the output gradient.

The local gradient is `y*(1-y)` (sigmoid derivative in terms of its output).
Both the `y[i]` AND the `(1 - y[i])` factors are required.

n=512; y values generated as `sigmoid(u)` for u in `[-4, 4]`.

**Tolerance:** fp32 elementwise (HVX qfloat path may differ by ~1 ULP).
Accepted if `|got - ref| <= 1e-4 + 1e-3 * |ref|` per element.

Use HVX fp32 multiply chain: compute `(1-y)` with `Q6_Vsf_vsub_VsfVsf`,
then fused multiply-multiply: `dy * y * (1-y)`.
