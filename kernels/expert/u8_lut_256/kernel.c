/*
 * u8_lut_256 — HVX 256-entry uint8->uint8 LUT (no clamping needed)
 *
 * Strategy:
 *   1. Preprocess each 128-byte half of the LUT with Q6_Vb_vshuff_Vb
 *      into the segment layout that vlut32 expects.
 *   2. 8 passes of vlut32/vlut32or with segments 0-7 cover all 256 entries.
 *      The index is in[i] directly (unsigned, 0..255); no reinterpretation needed.
 *   3. Handle the tail (n=1000 → tail=104 bytes) with a predicated store.
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *in, uint8_t *out, int n, const uint8_t *lut)
{
    /*
     * Preprocess the LUT using vshuff (bilateral SDK pattern).
     * sTab0: lut[0..127]  → segments 0-3
     * sTab1: lut[128..255] → segments 4-7
     */
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut));
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut + 128));

    int vlen = 128;
    int nvec = n / vlen;
    int tail = n % vlen;

    const uint8_t *src = in;
    uint8_t       *dst = out;

    for (int v = 0; v < nvec; v++) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        /* 8-pass 256-entry LUT lookup. */
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR(vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);

        *(HVX_Vector *)dst = res;
        src += vlen;
        dst += vlen;
    }

    /* Tail: n=1000 → tail = 104 bytes. */
    if (tail > 0) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR(vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);

        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail);
        Q6_vmem_QRIV(mask, (HVX_Vector *)dst, res);
    }
}
