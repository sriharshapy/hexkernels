# FP16 ReLU6 Activation (n=1000)

Implement:
```c
typedef __fp16 hvx_hf;
void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n);
```

Semantics (EXACT):
```
out[i] = (hvx_hf)(fminf(fmaxf((float)x[i], 0.0f), 6.0f));
```

Clamp each element to **[0, 6]**. Both bounds are exact fp16 constants —
result must be **bit-exact** to the scalar reference.

- n=1000; handle tail (n is not necessarily a multiple of 128).
- Use HVX fp16 intrinsics (`Q6_Vhf_vmax_VhfVhf` / `Q6_Vhf_vmin_VhfVhf`) for vectorized throughput.
- No HMX needed; this is a plain HVX fp16 elementwise operation.
