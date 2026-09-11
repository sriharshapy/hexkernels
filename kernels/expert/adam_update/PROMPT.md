# Adam Optimizer In-Place Update (n=256)

Implement:
```c
void candidate_kernel(float *w, float *m, float *v, const float *grad, int n,
                      float lr, float b1, float b2, float eps, int t);
```

**Semantics:** In-place Adam update. All three buffers `w`, `m`, and `v` are
updated in place:

```
m[i] = b1*m[i] + (1-b1)*grad[i]
v[i] = b2*v[i] + (1-b2)*grad[i]*grad[i]
mhat = m[i] / (1 - b1^t)
vhat = v[i] / (1 - b2^t)
w[i] = w[i] - lr * mhat / (sqrtf(vhat) + eps)
```

n=256, lr=0.001f, b1=0.9f, b2=0.999f, eps=1e-8f, t=10. The harness verifies
**all three** updated arrays (`m`, `v`, `w`) against the scalar reference.

The bias-correction denominators `(1 - b1^t)` and `(1 - b2^t)` are constants
for a given `t` -- compute them once before the loop.

Handle the tail when n is not a multiple of 32.

Use HVX fp32 intrinsics. Key operations: fused multiply-add for moment updates,
reciprocal sqrt (`Q6_Vsf_equals_Vsf` + `Q6_Vsf_vrsqrt_Vsf`) for the
`1/sqrt(vhat+eps)` term, vectorized multiply-subtract for the weight update.
