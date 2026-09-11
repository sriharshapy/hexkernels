/* Near-miss: correct requantization but skips the tanh LUT activation.
   Outputs the raw pre-activation int8 value instead of lut[pre_act].
   Fails because reference applies tanh_lut and the harness sweeps 2 distinct tables. */
#include <stdint.h>

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
        int32_t acc = b[j];
        for (int k = 0; k < I; k++)
            acc += (int32_t)Wx[j*I+k] * (int32_t)x[k];
        for (int k = 0; k < H; k++)
            acc += (int32_t)Wh[j*H+k] * (int32_t)h_prev[k];

        int64_t v    = (int64_t)acc * (int64_t)mult;
        int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
        r += zp;
        if (r >  127) r =  127;
        if (r < -128) r = -128;

        /* BUG: returns pre-activation instead of tanh_lut[pre_act] */
        h_t[j] = (int8_t)r;
    }
}
