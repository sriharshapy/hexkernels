/* NEAR-MISS: uses Iu1=1 instead of Iu1=0. This selects a different,
 * off-by-one alignment of the pattern against the block-matching
 * network -- a plausible flag mistake. Compiles fine but differs from
 * the correct group SAD values at every 128B boundary. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, uint32_t *out, int g) {
    uint32_t rt = 10u | (20u << 8) | (30u << 16) | (40u << 24);
    int nvec = g / 32;
    int i;
    for (i = 0; i < nvec; i++) {
        HVX_Vector va = *(const HVX_Vector *)(a + i * 128);
        HVX_VectorPair vp = Q6_W_vcombine_VV(va, va);
        HVX_VectorPair sad = Q6_Wuw_vrsad_WubRubI(vp, rt, 1);
        *(HVX_Vector *)(out + i * 32) = Q6_V_lo_W(sad);
    }
    for (int k = nvec * 32; k < g; k++) {
        int pattern[4] = {10, 20, 30, 40};
        uint32_t s = 0;
        for (int j = 0; j < 4; j++) {
            int d = (int)a[4*k+j] - pattern[j];
            s += (uint32_t)(d < 0 ? -d : d);
        }
        out[k] = s;
    }
}
