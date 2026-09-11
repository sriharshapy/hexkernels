/* NEAR-MISS: only covers vlut32 segments 0-6 (forgets segment 7). Compiles
 * and passes for indices in [0,223], but silently returns 0 for any index
 * in [224,255] (segment 7's 32-entry block is never OR'd in). Fails
 * bit-exact whenever the input touches that range (e.g. the pinned
 * idx=255 tail edge case). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, int8_t *out, int n, const int8_t *lut) {
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut));
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut + 128));

    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector vidx = *(const HVX_Vector *)(a + i);
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        /* MISSING: segment 7 (idx 224..255) never OR'd in -> those lanes stay 0 */
        *(HVX_Vector *)(out + i) = res;
    }
    for (; i < n; i++) {
        uint8_t idx = (uint8_t)a[i];
        out[i] = (idx >= 224) ? 0 : lut[idx];   /* mirror the same bug in the tail */
    }
}
