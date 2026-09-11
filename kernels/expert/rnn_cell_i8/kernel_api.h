#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>

/*
 * Vanilla Elman RNN cell (integer, one time-step, batch=1).
 *
 * Equation (all integer, bit-exact):
 *   acc[j] = dot(Wx[j,:], x, I) + dot(Wh[j,:], h_prev, H) + b[j]  (int32)
 *   pre_act[j] = requant(acc[j], mult, shift, zp)                    (int8)
 *   h_t[j]     = tanh_lut[(uint8_t)pre_act[j]]                      (int8)
 *
 * requant: round-half-away-from-zero, saturate to [-128,127]
 *   v    = (int64_t)acc * mult
 *   half = shift > 0 ? (1LL << (shift-1)) : 0
 *   r    = v>=0 ? (v+half)>>shift : -(((-v)+half)>>shift)
 *   r   += zp; clamp to [-128,127]
 *
 * Inputs:
 *   Wx      [H x I] row-major int8 weight matrix (input -> hidden)
 *   Wh      [H x H] row-major int8 weight matrix (hidden -> hidden)
 *   b       [H]     int32 bias
 *   x       [I]     int8 input
 *   h_prev  [H]     int8 previous hidden state
 *   h_t     [H]     int8 output (new hidden state)
 *   H, I    hidden / input dimensions (runtime)
 *   mult, shift, zp  requantization params (runtime, do NOT hardcode)
 *   tanh_lut  [256] int8 activation table indexed as (uint8_t)pre_act[j]
 */
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
#endif /* KERNEL_API_H */
