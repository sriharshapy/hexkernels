#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>

/*
 * Integer GRU cell (one time-step, batch=1).
 *
 * Notation: H=hidden dim, I=input dim, C=I+H (concatenated).
 * Weights Wz,Wr,Wn each [H x C] row-major int8.
 * Biases  bz,br,bn each [H] int32.
 * All matvecs use the SAME requant params (mult,shift,zp) -- runtime, not hardcoded.
 * Two LUT tables (sig_lut for gates, tanh_lut for candidate) each 256-entry int8.
 *
 * Step-by-step (bit-exact reference):
 *
 *  1. concat = [x[0..I-1], h_prev[0..H-1]]  (conceptual, C = I+H elements)
 *
 *  2. Update gate z (uses sig_lut):
 *       accz[j] = bz[j] + dot(Wz[j,:], concat)   (int32)
 *       pre_z[j] = requant(accz[j], mult, shift, zp)  (int8)
 *       z[j]    = sig_lut[(uint8_t)pre_z[j]]          (int8, range [-128,127])
 *
 *  3. Reset gate r (uses sig_lut):
 *       accr[j] = br[j] + dot(Wr[j,:], concat)   (int32)
 *       pre_r[j] = requant(accr[j], mult, shift, zp)  (int8)
 *       r[j]    = sig_lut[(uint8_t)pre_r[j]]          (int8)
 *
 *  4. Gated recurrent input rh (element-wise, integer blend):
 *       rh[j] = clamp( ((int32_t)(r[j]+128) * (int32_t)h_prev[j] + 64) >> 7, -128, 127 )
 *       (r[j]+128 in [0,255] acts as unsigned blend weight)
 *
 *  5. Candidate hidden n (uses tanh_lut; input part = x, recurrent part = rh):
 *       accn[j] = bn[j] + dot(Wn[j, 0:I], x) + dot(Wn[j, I:I+H], rh)  (int32)
 *       pre_n[j] = requant(accn[j], mult, shift, zp)  (int8)
 *       n[j]    = tanh_lut[(uint8_t)pre_n[j]]          (int8)
 *
 *  6. Output (blend z and h_prev vs n, integer):
 *       h_t[j] = clamp( ((int32_t)(128-z[j]) * (int32_t)h_prev[j]
 *                       + (int32_t)(z[j]+128) * (int32_t)n[j] + 128) >> 8, -128, 127 )
 *       ((z[j]+128) in [0,255] is the update fraction)
 *
 * requant (same rule as i8_gemm_requant):
 *   v = (int64_t)acc * mult
 *   half = shift > 0 ? (1LL<<(shift-1)) : 0
 *   r = v>=0 ? (v+half)>>shift : -(((-v)+half)>>shift)
 *   r += zp; clamp to [-128,127]
 *
 * NOTE: Wz, Wr, Wn columns are ordered [x-part | h-part]:
 *   Wz[j, 0:I]   corresponds to x  (input columns)
 *   Wz[j, I:I+H] corresponds to h_prev (recurrent columns)
 * Same layout for Wr and Wn (Wn recurrent part multiplies rh, not h_prev).
 */
void candidate_kernel(
    const int8_t  *Wz,        /* [H x C] update gate weights   */
    const int8_t  *Wr,        /* [H x C] reset gate weights    */
    const int8_t  *Wn,        /* [H x C] candidate weights     */
    const int32_t *bz,        /* [H] update gate bias          */
    const int32_t *br,        /* [H] reset gate bias           */
    const int32_t *bn,        /* [H] candidate bias            */
    const int8_t  *x,         /* [I] input                     */
    const int8_t  *h_prev,    /* [H] previous hidden state     */
    int8_t        *h_t,       /* [H] output (new hidden state) */
    int            H,
    int            I,
    int32_t        mult,       /* requant mult (runtime)        */
    int            shift,      /* requant shift (runtime)       */
    int8_t         zp,         /* requant zero-point (runtime)  */
    const int8_t  *sig_lut,   /* [256] sigmoid LUT (runtime)   */
    const int8_t  *tanh_lut   /* [256] tanh LUT (runtime)      */
);
#endif /* KERNEL_API_H */
