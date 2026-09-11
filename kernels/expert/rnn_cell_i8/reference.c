#include <stdint.h>

/*
 * Scalar baseline: vanilla Elman RNN cell.
 * h_t[j] = tanh_lut[(uint8_t)requant(Wx[j,:]*x + Wh[j,:]*h_prev + b[j])]
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
) {
    for (int j = 0; j < H; j++) {
        /* Accumulate: Wx[j,:]*x */
        int32_t acc = b[j];
        for (int k = 0; k < I; k++)
            acc += (int32_t)Wx[j*I+k] * (int32_t)x[k];
        /* Accumulate: Wh[j,:]*h_prev */
        for (int k = 0; k < H; k++)
            acc += (int32_t)Wh[j*H+k] * (int32_t)h_prev[k];

        /* Requantize: round-half-away-from-zero, then saturate to int8 */
        int64_t v    = (int64_t)acc * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        int8_t pre_act = (int8_t)r;

        /* LUT activation: index as unsigned byte */
        h_t[j] = tanh_lut[(uint8_t)pre_act];
    }
}
