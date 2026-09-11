/* Near-miss: skips the SiLU nonlinearity entirely -- multiplies up[i]
 * directly by gate[i] instead of silu_lut[gate[i]+128]*up[i]. The classic
 * "forgot the gate activation" SwiGLU bug (mirrors
 * i8_ffn_swiglu_block/nearmiss_no_silu.c in spirit, for this smaller
 * standalone elementwise op). Fails wherever silu_lut[idx] != gate[i]. */
#include <stdint.h>
void candidate_kernel(const int8_t *gate, const int8_t *up, const int8_t *silu_lut,
                      int8_t *out, int N, int32_t scale_mult, int scale_shift) {
    (void)silu_lut; /* BUG: SiLU LUT ignored entirely */
    for (int i = 0; i < N; i++) {
        int32_t prod = (int32_t)gate[i] * (int32_t)up[i]; /* BUG: no SiLU activation */
        int64_t r    = (int64_t)prod * (int64_t)scale_mult;
        int64_t half = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
        int64_t q    = (r >= 0) ? ((r + half) >> scale_shift)
                                 : -(((-r) + half) >> scale_shift);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}
