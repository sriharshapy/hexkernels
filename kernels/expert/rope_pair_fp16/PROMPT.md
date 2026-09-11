Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const __fp16 *x, const __fp16 *cos, const __fp16 *sin,
                          __fp16 *out, int n_pairs);

Standalone rotary position embedding (RoPE) on interleaved element pairs.
x: [2*n_pairs] fp16, interleaved -- (x[2p], x[2p+1]) form rotation pair p.
cos, sin: [n_pairs] fp16 each -- per-pair rotation coefficients.
out: [2*n_pairs] fp16, same interleaved layout as x.

For each pair p in [0, n_pairs), let d=2*p, d2=2*p+1 (NO floating point
shortcuts -- do the multiply/add/sub in float, cast to fp16 only at the
output boundary):
```
out[d]  = (float)x[d]*(float)cos[p] - (float)x[d2]*(float)sin[p]
out[d2] = (float)x[d]*(float)sin[p] + (float)x[d2]*(float)cos[p]
```

n_pairs=100 (x has 200 fp16 elements -- NOT a multiple of 64 fp16
lanes-per-HVX-vector, a real tail path). cos/sin values lie in [-1,1];
x magnitudes vary. Includes hand-set edge pairs: p=0 with cos=1,sin=0
(identity -- out should equal x unchanged) and another pair with cos=0,
sin=1 (pure 90-degree rotate: out[d]=-x[d2], out[d2]=x[d]).

Output is compared to an independent float32 scalar reference with an
fp16 tolerance (HVX float arithmetic is non-IEEE qfloat, not bit-exact).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
