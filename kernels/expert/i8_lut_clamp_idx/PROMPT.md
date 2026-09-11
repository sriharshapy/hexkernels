Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int n,
                          const int8_t *lut, uint8_t lo, uint8_t hi);
Apply a 256-entry LUT with a CLAMPED index:
  1. Reinterpret: idx = (uint8_t)in[i]
  2. Clamp:       if (idx < lo) idx = lo;  if (idx > hi) idx = hi;
  3. Lookup:      out[i] = lut[idx]
lut has 256 entries. lo and hi are RUNTIME params (0 <= lo <= hi <= 255) —
do NOT hardcode them. The function is called with different lo/hi values.
n=1000 (not a multiple of 128 — handle the tail).
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
