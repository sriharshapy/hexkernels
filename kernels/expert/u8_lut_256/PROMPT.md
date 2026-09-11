Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *in, uint8_t *out, int n, const uint8_t *lut);
Apply a 256-entry uint8->uint8 lookup table: out[i] = lut[in[i]].
The input is UNSIGNED (values 0..255); the index is in[i] directly with NO reinterpretation.
lut has exactly 256 entries supplied as a runtime pointer (do NOT hardcode).
n=1000 (not a multiple of 128 — handle the tail). Prefer HVX table-lookup intrinsics
(Q6 vlut family, e.g. `Q6_Vb_vlut32_VbVbR` / `Q6_Vb_vlut32or_VbVbVbR`); include
the HVX headers and the HVX headers. Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
