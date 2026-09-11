Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int T, int L);

T=4 tensors, each of length L=256, stored contiguously: tensor t occupies
indices [t*L, t*L+L) in a, b, and out.

Compute out[t*L+i] = (int8_t)(a[t*L+i] + b[t*L+i]) for all t in [0,T) and i in [0,L).
Use two's-complement int8 wraparound (matching Q6_Vb_vadd_VbVb -- NOT saturating).
Total elements T*L = 1024, which is an exact multiple of the HVX vector width (128 bytes).

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
