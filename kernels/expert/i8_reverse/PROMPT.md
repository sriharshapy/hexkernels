Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const int8_t *in, int8_t *out, int n);

Reverse a 1D int8 array:
 out[i] = in[n-1-i] for i in [0, n)

n=501 (not a multiple of 128 — handle the tail). Pure data movement, no arithmetic.
HVX hint: `Q6_V_vdelta_VV` can reverse the bytes within a vector when every byte of
the control vector is 0x7F, or you can use `Q6_V_vror_VR` / `Q6_V_valign_VVR` for
reordering. Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
