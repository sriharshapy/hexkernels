Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *a, int8_t *out, int n,
                          int32_t mult, int shift, int8_t zp,
                          const int8_t lut[256]);
Requantize int32 to a uint8 index, then look up sigmoid in a 256-entry LUT.
For each i in [0, n):
  Step 1 -- requantize (round-half-away-from-zero):
    v    = (int64_t)a[i] * (int64_t)mult
    half = shift > 0 ? (1LL << (shift-1)) : 0
    r    = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
    r   += zp
  Step 2 -- clamp to uint8 range:  idx = clamp(r, 0, 255)
  Step 3 -- LUT lookup:            out[i] = lut[idx]
n=1024 -- handle the tail. mult, shift, zp, and lut[] are runtime params (do NOT hardcode).
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
