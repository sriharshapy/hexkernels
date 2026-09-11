/* NEAR-MISS: stores Q6_V_lo_W into out_lo and Q6_V_hi_W into out_hi --
 * the naming-intuitive but WRONG mapping (lo(pair) is pred?a:b, which
 * is out_hi by our spec, not out_lo). This swaps the two outputs
 * entirely. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out_hi, int8_t *out_lo, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        HVX_VectorPred qt = Q6_Q_vcmp_gt_VbVb(va, vb);
        HVX_VectorPair p = Q6_W_vswap_QVV(qt, va, vb);
        *(HVX_Vector *)(out_lo + i) = Q6_V_lo_W(p);   /* WRONG: swapped */
        *(HVX_Vector *)(out_hi + i) = Q6_V_hi_W(p);   /* WRONG: swapped */
    }
    for (; i < n; i++) {
        int pred = a[i] > b[i];
        out_lo[i] = pred ? a[i] : b[i];
        out_hi[i] = pred ? b[i] : a[i];
    }
}
