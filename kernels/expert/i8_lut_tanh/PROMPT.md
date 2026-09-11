Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut);

Apply a 256-entry int8->int8 lookup table representing the tanh activation:
    out[i] = lut[(uint8_t)in[i]]

The index is the input byte reinterpreted as UNSIGNED (in[i] in [-128,127] maps to
index in [0,255], i.e. -128->128, -1->255, 0->0, 127->127). lut has 256 entries.
n=1024 (a multiple of 128; still handle tails generically). The LUT is a runtime
parameter -- do NOT hardcode any table values.

Prefer HVX table-lookup intrinsics (Q6_Vb_vlut32_VbVbR or similar vlut family);
include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
