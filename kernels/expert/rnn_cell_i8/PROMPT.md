Implement ONLY this function (Hexagon HVX C):

    void candidate_kernel(
        const int8_t  *Wx,
        const int8_t  *Wh,
        const int32_t *b,
        const int8_t  *x,
        const int8_t  *h_prev,
        int8_t        *h_t,
        int            H,
        int            I,
        int32_t        mult,
        int            shift,
        int8_t         zp,
        const int8_t  *tanh_lut
    );

Compute one step of a vanilla Elman RNN cell (integer, batch=1).

Equation for each hidden unit j in [0, H):
  Step 1 -- accumulate (int32):
    acc[j] = b[j]
    acc[j] += dot(Wx[j*I : j*I+I], x[0:I])    // input projection
    acc[j] += dot(Wh[j*H : j*H+H], h_prev[0:H])// recurrent projection

  Step 2 -- requantize acc[j] to int8 (round-half-away-from-zero, then saturate):
    v      = (int64_t)acc[j] * (int64_t)mult
    half   = (shift > 0) ? (1LL << (shift-1)) : 0
    r      = (v >= 0) ? (v + half) >> shift : -(((-v) + half) >> shift)
    r     += zp
    pre_act = clamp(r, -128, 127)

  Step 3 -- lookup activation:
    h_t[j] = tanh_lut[(uint8_t)pre_act]   // index as unsigned byte [0..255]

Parameters:
  Wx      [H x I] int8 weight matrix, row-major (row j = Wx[j*I .. j*I+I-1])
  Wh      [H x H] int8 weight matrix, row-major
  b       [H]     int32 bias vector
  x       [I]     int8 input vector
  h_prev  [H]     int8 previous hidden state
  h_t     [H]     int8 output (new hidden state)
  H=32, I=32 (both multiples of 4, but NOT 128 -- handle tails)
  mult, shift, zp: runtime requantization params -- do NOT hardcode
  tanh_lut: 256-entry int8 LUT for the activation -- do NOT hardcode or ignore

Optimization hints:
  - Use Q6_Vw_vrmpyacc_VwVbVb (vrmpy) for the int8 dot products.
  - H=32 fits in one HVX register (32 int32 lanes or 128 int8 lanes).
  - Process the Wx and Wh matvecs in the same output-row loop.
  - For the LUT: use Q6_Vb_vlut32_VbVbI or scalar gather after requant.
  - mult and shift can be negative (handle signed arithmetic correctly).

Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
