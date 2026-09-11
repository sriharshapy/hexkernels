# SGD with Momentum In-Place Update (n=512)

Implement:
```c
void candidate_kernel(float *w, float *v, const float *grad, int n,
 float lr, float mu);
```

**Semantics:** In-place SGD with momentum. Both `w` (weights) and `v` (velocity
buffer) are updated:

```
v[i] = mu * v[i] + grad[i]
w[i] = w[i] - lr * v[i]
```

n=512, lr=0.01f, mu=0.9f. The harness verifies **both** the updated `w` and `v`
arrays against the scalar reference.

Handle the tail when n is not a multiple of 32.

Use HVX fp32 intrinsics for vectorized throughput. Each HVX vector holds 32 fp32
values (128 bytes / 4 bytes each). The velocity update and weight update can be
fused in a single pass over the three arrays.
