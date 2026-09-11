/* expert: HVX Q15 fractional multiply-high via Q6_Vh_vmpy_VhVh_s1_rnd_sat. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vmpy_VhVh_s1_rnd_sat(va, vb);
    }
    for (; i < n; i++) {
        int32_t P = (int32_t)a[i] * (int32_t)b[i];
        int32_t bias = (P >= 0) ? 16384 : -16384;
        int32_t q = (P + bias) / 32768;
        if (q > 32767) q = 32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
