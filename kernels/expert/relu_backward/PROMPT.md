# ReLU Backward Pass (n=512)

Implement:
```c
void candidate_kernel(const float *x, const float *dy, float *dx, int n);
```

**Semantics:** `dx[i] = (x[i] > 0.0f) ? dy[i] : 0.0f`

ReLU backward (gradient) pass: `x` is the forward-pass input, `dy` is the
upstream gradient, `dx` is the output gradient. Gate is determined by the
SIGN OF `x`, not `dy`.

n=512; inputs span negative and positive values in `[-2, 2]`.
At `x[i] == 0`, use 0 (right-sided subgradient convention).

**Tolerance:** fp32 elementwise (HVX qfloat path may differ by ~1 ULP).
Accepted if `|got - ref| <= 1e-4 + 1e-3 * |ref|` per element.

Use HVX compare (`Q6_Q_vcmp_gtf_VsfVsf`) to build a boolean mask from x,
then select (`Q6_Vsf_vmux_QVsfVsf`) between dy and zero using that mask.
