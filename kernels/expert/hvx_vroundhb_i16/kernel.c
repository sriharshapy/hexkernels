/* expert: HVX round-and-narrow via Q6_Vb_vround_VhVh_sat(Vu=a,Vv=b).
 * Directly produces the interleaved (b-even, a-odd) byte output per
 * 64-pair chunk -- no manual interleave needed, the hardware does it. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
static int round_div256(int16_t x) { return ((int)x + 128) >> 8; }

void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + 2*i) = Q6_Vb_vround_VhVh_sat(va, vb);
    }
    for (; i < n; i++) {
        out[2*i]     = sat8(round_div256(b[i]));
        out[2*i + 1] = sat8(round_div256(a[i]));
    }
}
