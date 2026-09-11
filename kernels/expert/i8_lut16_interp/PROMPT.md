Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut);
Apply a 16-entry linearly-interpolated lookup table:
  For each element, reinterpret in[i] as uint8_t to get b.
  hi = b >> 4          (table index, 0..15)
  lo = b & 0xF         (interpolation fraction, 0..15)
  out[i] = (int8_t)(lut[hi] + (((int16_t)(lut[hi+1] - lut[hi]) * lo) >> 4))
The shift >>4 is TRUNCATING (C right-shift of signed int16_t, not rounding).
lut has 17 entries (lut[0]..lut[16]) so lut[hi+1] is always valid.
lut is supplied as a runtime pointer — do NOT hardcode its values.
n=1000 (not a multiple of 128 — handle the tail).
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
