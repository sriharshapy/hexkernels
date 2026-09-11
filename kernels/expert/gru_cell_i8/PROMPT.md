Implement ONLY this function (Hexagon HVX C):

 void candidate_kernel(
 const int8_t *Wz,
 const int8_t *Wr,
 const int8_t *Wn,
 const int32_t *bz,
 const int32_t *br,
 const int32_t *bn,
 const int8_t *x,
 const int8_t *h_prev,
 int8_t *h_t,
 int H,
 int I,
 int32_t mult,
 int shift,
 int8_t zp,
 const int8_t *sig_lut,
 const int8_t *tanh_lut
 );

Compute one step of an integer GRU cell (batch=1). H=24, I=24, C=I+H=48.
Weights Wz, Wr, Wn are each [H x C] row-major int8 (columns 0..I-1 = input part,
columns I..I+H-1 = recurrent part). Biases bz, br, bn are [H] int32.
All three matvecs share the same requant params (mult, shift, zp) -- do NOT hardcode.

Bit-exact steps:

 Step 1 -- Update gate z (uses sig_lut):
 accz[j] = bz[j] + dot(Wz[j,0:I], x) + dot(Wz[j,I:I+H], h_prev) (int32)
 pre_z[j] = requant(accz[j], mult, shift, zp) (int8)
 z[j] = sig_lut[(uint8_t)pre_z[j]] (int8)

 Step 2 -- Reset gate r (uses sig_lut):
 accr[j] = br[j] + dot(Wr[j,0:I], x) + dot(Wr[j,I:I+H], h_prev) (int32)
 pre_r[j] = requant(accr[j], mult, shift, zp) (int8)
 r[j] = sig_lut[(uint8_t)pre_r[j]] (int8)

 Step 3 -- Gated recurrent input rh (integer, element-wise):
 rh[j] = clamp( ((int32_t)(r[j]+128) * (int32_t)h_prev[j] + 64) >> 7, -128, 127 )
 (r[j]+128 is in [0,255] and acts as an unsigned blend weight)

 Step 4 -- Candidate hidden n (uses tanh_lut; recurrent part uses rh):
 accn[j] = bn[j] + dot(Wn[j,0:I], x) + dot(Wn[j,I:I+H], rh) (int32)
 pre_n[j] = requant(accn[j], mult, shift, zp) (int8)
 n[j] = tanh_lut[(uint8_t)pre_n[j]] (int8)

 Step 5 -- Output blend (z is the update fraction):
 h_t[j] = clamp( ((int32_t)(128-z[j]) * (int32_t)h_prev[j]
 + (int32_t)(z[j]+128) * (int32_t)n[j] + 128) >> 8, -128, 127 )

requant rule (same as i8_gemm_requant, round-half-away-from-zero):
 v = (int64_t)acc * mult
 half = shift > 0 ? (1LL << (shift-1)) : 0
 r = v>=0 ? (v+half)>>shift : -(((-v)+half)>>shift)
 r += zp; clamp to [-128, 127]

IMPORTANT: the gate order is z THEN r (wrong order = wrong answer). The recurrent part
of the candidate (Wn) uses rh (r*h_prev blend), NOT h_prev directly.

mult, shift, zp are runtime params -- do NOT hardcode. sig_lut and tanh_lut are
runtime tables -- do NOT hardcode or confuse them (wrong table = wrong answer).

Optimization hints:
 - C=48 > 32, handle 2 x 32-column blocks in the matvec with possible tail.
 - Use vrmpy (`Q6_Vw_vrmpy_VbVb`) for int8 dot products.
 - The three gates can share the same weight-load inner loop structure.
 - For LUT: vlut32 or scalar gather after requant.

Respond with a single complete C code block and CLOSE the fence with ```.
