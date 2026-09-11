Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, int8_t *out, int n, const int8_t *lut);

Compute a quantized SiLU (x*sigmoid(x)) activation via a 256-entry DIRECT-indexed
lookup table gather.
Semantics: idx = (uint8_t)x[i]  (the signed byte's raw bit pattern reinterpreted as an
unsigned index in [0,255] -- NOT a scale/shift index derivation, just a direct cast);
out[i] = lut[idx].

`lut` is a runtime 256-entry int8 table (already dequantized/SiLU'd/requantized by the
harness at a fixed scale) -- read it at runtime, do NOT hardcode any activation math or
table values inside candidate_kernel. This is a pure gather.

n=1000. n is NOT a multiple of 128 -- handle the tail.
Prefer HVX vector intrinsics (e.g. the vlut32 256-entry byte-LUT gather idiom); include
<hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
