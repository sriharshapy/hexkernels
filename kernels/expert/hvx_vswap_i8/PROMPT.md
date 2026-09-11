# HVX Drill: Per-Lane Conditional Swap-Select, vswap (n=1005)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out_hi, int8_t *out_lo, int n);
```

Semantics: let pred[i] = (a[i] > b[i]) (signed compare). For i in [0, n):
```
out_hi[i] = pred[i] ? a[i] : b[i]
out_lo[i] = pred[i] ? b[i] : a[i]
```

Use the HVX intrinsic `Q6_Q_vcmp_gt_VbVb(Vu, Vv)` to build the predicate
from a and b, then `Q6_W_vswap_QVV(Qt, Vu, Vv)` to produce both outputs
at once as a VectorPair: `Q6_V_lo_W` of that pair gives `pred?Vu:Vv`, and
`Q6_V_hi_W` gives the complementary `pred?Vv:Vu`. With Vu=a, Vv=b, store
`Q6_V_lo_W(pair)` into `out_hi` and `Q6_V_hi_W(pair)` into `out_lo`.

- n=1005 (not a multiple of the 128-byte HVX vector width) -- handle the
  scalar tail (this op is per-lane, so the tail is a straightforward
  elementwise fallback, unlike the block-permutation ops).
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
