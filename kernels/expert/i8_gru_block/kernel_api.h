#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>

/*
 * MULTI-STEP integer GRU block (T timesteps, batch=1), with a FUSED update/reset
 * gate matmul: distinct from gru_cell_i8 (which is a SINGLE step with THREE
 * separate weight matrices Wz, Wr, Wn). Here:
 *   - Wzr [2H x C] stacks the update-gate rows [0,H) then the reset-gate rows
 *     [H,2H) -- ONE weight tensor and ONE loop compute BOTH gates' pre-activations
 *     (they share the exact same input concat[x_t | h_prev]).
 *   - Wn [H x C] is the candidate-hidden weight (recurrent part multiplies rh,
 *     not h_prev, so it cannot be fused into the same matmul as z/r).
 *   - The cell recurs over T timesteps, feeding h_t back in as h_prev for step
 *     t+1, and writes EVERY timestep's hidden state to h_out.
 *
 * Notation: H=hidden dim, I=input dim per step, C=I+H (concatenated).
 * All matvecs use the SAME requant params (mult,shift,zp) -- runtime, not
 * hardcoded. Two LUT tables (sig_lut for gates, tanh_lut for candidate),
 * each 256-entry int8.
 *
 * Step-by-step (bit-exact reference), for t in [0,T):
 *
 *  1. concat_xh = [x[t*I+0..t*I+I-1], h_prev[0..H-1]]   (C = I+H elements)
 *
 *  2. Fused update/reset gates (rows 0..H-1 = z, rows H..2H-1 = r):
 *       for j in [0,2H):
 *         acc[j]  = bzr[j] + dot(Wzr[j,:], concat_xh)          (int32)
 *         q[j]    = requant(acc[j], mult, shift, zp)           (int8)
 *       z[j] = sig_lut[(uint8_t)q[j]]        for j in [0,H)
 *       r[j] = sig_lut[(uint8_t)q[H+j]]      for j in [0,H)
 *
 *  3. Gated recurrent input rh (element-wise, integer blend):
 *       rh[j] = clamp( ((int32_t)(r[j]+128) * (int32_t)h_prev[j] + 64) >> 7, -128, 127 )
 *
 *  4. Candidate hidden n (recurrent part uses rh, not h_prev):
 *       concat_xrh = [x[t*I+0..t*I+I-1], rh[0..H-1]]
 *       accn[j]  = bn[j] + dot(Wn[j,:], concat_xrh)            (int32)
 *       pre_n[j] = requant(accn[j], mult, shift, zp)           (int8)
 *       n[j]     = tanh_lut[(uint8_t)pre_n[j]]                 (int8)
 *
 *  5. Output (blend z and h_prev vs n, integer):
 *       h_t[j] = clamp( ((int32_t)(128-z[j]) * (int32_t)h_prev[j]
 *                       + (int32_t)(z[j]+128) * (int32_t)n[j] + 128) >> 8, -128, 127 )
 *
 *  6. h_out[t*H .. t*H+H-1] = h_t[:];  h_prev = h_t   (propagate to next step)
 *
 * requant (same rule as gru_cell_i8 / i8_gemm_requant):
 *   v = (int64_t)acc * mult
 *   half = shift > 0 ? (1LL<<(shift-1)) : 0
 *   r = v>=0 ? (v+half)>>shift : -(((-v)+half)>>shift)
 *   r += zp; clamp to [-128,127]
 *
 * Wzr, Wn columns are ordered [x-part | h-part]: Wzr[j,0:I] / Wn[j,0:I]
 * correspond to x_t (input columns); Wzr[j,I:I+H] corresponds to h_prev
 * (recurrent columns, update/reset gates); Wn[j,I:I+H] corresponds to rh
 * (candidate's recurrent columns, NOT h_prev).
 */
void candidate_kernel(
    const int8_t  *Wzr,       /* [2H x C] fused update+reset gate weights */
    const int8_t  *Wn,        /* [H x C] candidate weights                */
    const int32_t *bzr,       /* [2H] fused update+reset gate bias        */
    const int32_t *bn,        /* [H] candidate bias                       */
    const int8_t  *x,         /* [T x I] input sequence                   */
    const int8_t  *h0,        /* [H] initial hidden state                 */
    int8_t        *h_out,     /* [T x H] output (every step's hidden state) */
    int            T,
    int            H,
    int            I,
    int32_t        mult,      /* requant mult (runtime)                   */
    int            shift,     /* requant shift (runtime)                  */
    int8_t         zp,        /* requant zero-point (runtime)              */
    const int8_t  *sig_lut,   /* [256] sigmoid LUT (runtime)               */
    const int8_t  *tanh_lut   /* [256] tanh LUT (runtime)                  */
);
#endif /* KERNEL_API_H */
