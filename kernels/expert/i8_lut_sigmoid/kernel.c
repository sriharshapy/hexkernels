/* EXPERT (achievability bar) — HVX 256-entry byte LUT via vlut32.
 * index = (uint8_t)in[i] in [0,255]. Preprocess the 256-entry LUT into two
 * shuffled 128-byte halves; 8 vlut32/vlut32or passes (segments 0-7) cover all
 * 256 entries (segments 0-3 -> lut[0..127], 4-7 -> lut[128..255]). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut) {
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut));
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut + 128));

    const int vlen = 128;
    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        HVX_Vector vidx = *(const HVX_Vector *)(in + i);
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);
        *(HVX_Vector *)(out + i) = res;
    }
    int tail = n - i;
    if (tail > 0) {
        HVX_Vector vidx = *(const HVX_Vector *)(in + i);
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);
        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail);
        Q6_vmem_QRIV(mask, (HVX_Vector *)(out + i), res);
    }
}
