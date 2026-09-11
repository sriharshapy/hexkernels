/* expert: HVX saturating int16 subtract via Q6_Vh_vsub_VhVh_sat. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int16_t sat16(int v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

void candidate_kernel(const int16_t *a, const int16_t *b, int16_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vh_vsub_VhVh_sat(va, vb);
    }
    for (; i < n; i++)
        out[i] = sat16((int)a[i] - (int)b[i]);
}
