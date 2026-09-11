Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n);
Compute out[i] = a[i] < b[i] ? a[i] : b[i] for i in [0, n) using SIGNED int8 comparison
(matching Q6_Vb_vmin_VbVb; do NOT treat the bytes as unsigned). n=500000 (working set ~1.5 MB, DDR-bound); n is not a multiple of 128 — handle the tail.
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
