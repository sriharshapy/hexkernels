/* expert: HVX saturating narrow via Q6_Vb_vpack_VhVh_sat(Vu=a,Vv=b).
 * Full chunks: one vpack call produces the whole 128-byte concatenated
 * (b-then-a) result directly. Tail (m<64): plain scalar, matching the
 * same chunked formula (the hardware concatenation boundary would land
 * mid-vector for a partial chunk, so scalar is simplest/safest here). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}

void candidate_kernel(const int16_t *a, const int16_t *b, int8_t *out, int n) {
    int base = 0;
    for (; base + 64 <= n; base += 64) {
        HVX_Vector va = *(const HVX_Vector *)(a + base);
        HVX_Vector vb = *(const HVX_Vector *)(b + base);
        *(HVX_Vector *)(out + 2*base) = Q6_Vb_vpack_VhVh_sat(va, vb);
    }
    int m = n - base;
    for (int j = 0; j < m; j++) {
        out[2*base + j]     = sat8(b[base + j]);
        out[2*base + m + j] = sat8(a[base + j]);
    }
}
