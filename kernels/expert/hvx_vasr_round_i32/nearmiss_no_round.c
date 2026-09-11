/* NEAR-MISS: uses the non-rounding Q6_Vh_vasr_VwVwR_sat (truncating shift,
 * no +bias before the shift) instead of Q6_Vh_vasr_VwVwR_rnd_sat. Compiles
 * and passes whenever the shifted-out bits are exactly zero, but fails on
 * the pinned rounding edge cases (e.g. lo=16,shift=5 should round to 1,
 * truncating gives 0). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *lo, const int32_t *hi, int16_t *out, int n, int shift) {
    int nvec = n / 32;
    int i;
    for (i = 0; i < nvec; i++) {
        HVX_Vector vlo = *(const HVX_Vector *)(lo + i * 32);
        HVX_Vector vhi = *(const HVX_Vector *)(hi + i * 32);
        *(HVX_Vector *)(out + i * 64) = Q6_Vh_vasr_VwVwR_sat(vhi, vlo, shift);   /* WRONG: no rounding */
    }
    for (int k = nvec * 32; k < n; k++) {
        int32_t rl = lo[k] >> shift;   /* WRONG: no rounding bias */
        int32_t rh = hi[k] >> shift;   /* WRONG: no rounding bias */
        if (rl > 32767) rl = 32767;  if (rl < -32768) rl = -32768;
        if (rh > 32767) rh = 32767;  if (rh < -32768) rh = -32768;
        out[2*k]   = (int16_t)rl;
        out[2*k+1] = (int16_t)rh;
    }
}
