Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int n, uint8_t thresh);
Inverse threshold, STRICT: out[i] = (in[i] < thresh) ? 255 : 0. When in[i] == thresh the output
is 0 (strict less-than, NOT less-than-or-equal). n=1000 (not a multiple of 128 — handle the tail).
Prefer HVX compare + select intrinsics (Q6_Q_vcmp_gt + Q6_V_vmux); include <hexagon_types.h>
and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
