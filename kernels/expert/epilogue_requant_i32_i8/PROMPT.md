Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *acc, int8_t *out, int n, int32_t mult, int shift);
For each i, compute:
    v      = (int64_t)acc[i] * (int64_t)mult
    half   = shift > 0 ? (1LL << (shift-1)) : 0
    r      = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)   // round HALF AWAY FROM ZERO
    out[i] = (int8_t) clamp(r, -128, 127)
Apply `mult` in 64-bit BEFORE the shift. `mult` and `shift` are RUNTIME parameters
(the harness sweeps at least 2 distinct (mult, shift) pairs -- do not hardcode
either). Pinned size: n=1000 -- n is not a multiple of the natural 32-lane int32
vector width, so handle the tail. Use HVX intrinsics for the bulk of the work;
include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
