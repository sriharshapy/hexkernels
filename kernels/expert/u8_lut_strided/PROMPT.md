Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int n,
                          const uint8_t *lut, int k);
Apply a 256-entry uint8->uint8 LUT to every k-th element; copy all others UNCHANGED:
  for i in [0, n):
    if (i % k == 0):  out[i] = lut[in[i]]
    else:             out[i] = in[i]
k is a RUNTIME param (do NOT hardcode k). The function is called with different k values.
lut has 256 entries, supplied as a runtime pointer (do NOT hardcode).
n=1000 (not a multiple of 128 — handle the tail).
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
