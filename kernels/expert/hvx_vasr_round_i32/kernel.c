/* expert: same as solutions/s1.c (drill task, accelerable=false; 1.0x is fine). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *lo, const int32_t *hi, int16_t *out, int n, int shift) {
    int nvec = n / 32;
    int i;
    for (i = 0; i < nvec; i++) {
        HVX_Vector vlo = *(const HVX_Vector *)(lo + i * 32);
        HVX_Vector vhi = *(const HVX_Vector *)(hi + i * 32);
        *(HVX_Vector *)(out + i * 64) = Q6_Vh_vasr_VwVwR_rnd_sat(vhi, vlo, shift);
    }
    for (int k = nvec * 32; k < n; k++) {
        int32_t rl = (shift == 0) ? lo[k] : ((lo[k] + (1 << (shift - 1))) >> shift);
        int32_t rh = (shift == 0) ? hi[k] : ((hi[k] + (1 << (shift - 1))) >> shift);
        if (rl > 32767) rl = 32767;  if (rl < -32768) rl = -32768;
        if (rh > 32767) rh = 32767;  if (rh < -32768) rh = -32768;
        out[2*k]   = (int16_t)rl;
        out[2*k+1] = (int16_t)rh;
    }
}
