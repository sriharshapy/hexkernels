Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const float *x, float *y, int T, int L, float alpha);

T=4 tensors, each of length L=256, stored contiguously: tensor t occupies
indices [t*L, t*L+L) in x (read-only) and y (in-place update).

Semantics (in-place AXPY):
    y[t*L+i] = alpha * x[t*L+i] + y[t*L+i]   for all t in [0,T), i in [0,L).
alpha = 0.5f.  Total elements T*L = 1024, an exact multiple of the HVX vector width.

Use HVX float intrinsics. NOTE: `Q6_Vsf_vmpy_VsfVsf` fails to select at v68 — do the
multiply in qf32 (`Q6_Vqf32_vmpy_Vsf...`/`Q6_Vqf32_vadd_...`) and convert back with
`Q6_Vsf_equals_Vqf32`, or fold alpha in via qf32; `Q6_Vsf_vadd_VsfVsf` is fine.
Comparison uses a tight abs+rel tolerance (not bit-exact) because HVX qfloat differs
slightly from IEEE scalar -- just implement the correct fused multiply-add semantics.

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
