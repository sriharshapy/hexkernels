Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *gate, const int8_t *up, const int8_t *silu_lut,
                          int8_t *out, int N, int32_t scale_mult, int scale_shift);
Standalone SwiGLU gating step (elementwise -- this is only the gating nonlinearity,
NOT the surrounding matmuls; gate/up are already-projected 1D arrays):
  Step 1 -- LUT:      idx    = (uint8_t)((int)gate[i] + 128)          (in [0,255])
                       g      = silu_lut[idx]                            (int8, runtime)
  Step 2 -- multiply:  prod   = (int32_t)g * (int32_t)up[i]              (int32)
  Step 3 -- requant (round-half-away-from-zero):
                       r    = (int64_t)prod * (int64_t)scale_mult
                       half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
                       q    = (r >= 0) ? ((r + half) >> scale_shift) : -((-r + half) >> scale_shift)
                       out[i] = clamp(q, -128, 127)
gate and up are [N] int8 (1D arrays, already-projected gate/up values).
silu_lut is [256] int8; index into it as (uint8_t)(gate[i] + 128). It is a RUNTIME
  parameter (represents SiLU of the dequantized gate, re-quantized) -- do NOT hardcode it.
scale_mult (int32, positive) and scale_shift (int, >=0) are RUNTIME parameters swept
  by the harness across multiple (mult, shift) pairs -- do NOT hardcode them.
N=150 (1D length). N is NOT a multiple of 128 -- one full 128-lane vector block plus a
  22-element tail.
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
