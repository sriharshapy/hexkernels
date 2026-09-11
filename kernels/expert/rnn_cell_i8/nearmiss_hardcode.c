/* Near-miss: hardcodes mult=3, shift=7, zp=0 and ignores tanh_lut entirely
   (uses a fixed identity mapping). Fails when other (mult,shift,zp) sweep sets
   or the second LUT table is active. */
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

        /* Hardcoded mult=3, shift=7 -- ignores runtime params */
        long long v = (long long)acc * 3;
        long long h = 64; /* 1 << (7-1) */
        long long r = (v >= 0) ? ((v + h) >> 7) : -(((-v) + h) >> 7);
        r += zp; /* at least uses zp */
        if (r >  127) r =  127;
        if (r < -128) r = -128;

        /* Hardcoded identity -- ignores tanh_lut */
        h_t[j] = (int8_t)r;
    }
}
