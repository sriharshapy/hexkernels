Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, const uint8_t *mask,
                          int8_t *out, int n);
Conditional select (where): out[i] = mask[i] ? a[i] : b[i].
mask[i] is nonzero-true: ANY non-zero value selects a[i]; only mask[i]==0 selects b[i].
n=1000 (not a multiple of 128 — handle the tail). Prefer HVX select intrinsics
(Q6_Q_vcmp_eq_VubVub, Q6_V_vmux); include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
