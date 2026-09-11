# FP32 ReLU6 Activation (n=1000)

Implement:
```c
void candidate_kernel(const float *x, float *out, int n);
```

Semantics (EXACT):
```
out[i] = fminf(fmaxf(x[i], 0.0f), 6.0f);
```

Clamp each element to **[0, 6]**. Both bounds are exact fp32 constants --
result must be **bit-exact** to the scalar reference.

- n=1000; handle tail (n is not necessarily a multiple of 128).
- Use HVX float intrinsics (`Q6_Vsf_*`) for vectorized throughput.
- Include `<hexagon_types.h>` and `<hexagon_protos.h>` for HVX intrinsics.
- No HMX needed; this is a plain HVX fp32 elementwise operation.
- Include `<math.h>` for scalar fallback/tail handling if needed.

Hint: `Q6_Vsf_vmax_VsfVsf` / `Q6_Vsf_vmin_VsfVsf` implement the clamp.
Each HVX vector holds 32 fp32 values (128 bytes / 4 bytes each).
