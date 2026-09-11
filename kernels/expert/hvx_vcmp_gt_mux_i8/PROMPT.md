# HVX Drill: Compare + 3-Input Select (n=1013)

Implement ONLY this function (Hexagon HVX C):
```c
void candidate_kernel(const int8_t *a, const int8_t *b, const int8_t *c, int8_t *out, int n);
```

Semantics: `out[i] = (a[i] > b[i]) ? a[i] : c[i]` for `i` in `[0, n)`
(signed compare). Note the false branch selects `c`, NOT `b` -- this is
a genuine 3-input select, not a 2-input max.

Use `Q6_Q_vcmp_gt_VbVb(a, b)` to build the predicate vector, then
`Q6_V_vmux_QVV(Qt, a, c)` to select between `a` and `c` based on it.

- n=1013 (not a multiple of the 128-byte HVX vector width) -- handle the scalar tail.
- Arrays are 128-byte aligned (HVX_ALIGN).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block.
