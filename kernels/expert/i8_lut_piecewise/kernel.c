/*
 * i8_lut_piecewise — HVX dual-table 256-entry LUT
 *
 * Strategy:
 *   The piecewise rule maps each int8 input through one of two 128-entry tables:
 *     in[i] < 0  (uint8 = 128..255): out[i] = tableA[(uint8_t)in[i] - 128]
 *     in[i] >= 0 (uint8 = 0..127):   out[i] = tableB[in[i]]
 *
 *   We merge both 128-entry tables into a single 256-entry combined table:
 *     combined[j]       = tableB[j]    for j = 0..127   (non-negative inputs)
 *     combined[128 + j] = tableA[j]    for j = 0..127   (negative inputs)
 *
 *   Then the raw unsigned reinterpretation of each int8 input byte is already
 *   the correct index into combined[]:
 *     - non-negative x: uint8(x) in 0..127   → combined[uint8(x)] = tableB[x]    ✓
 *     - negative x:     uint8(x) in 128..255 → combined[uint8(x)] = tableA[uint8(x)-128] ✓
 *
 *   Apply the standard 8-pass Q6_Vb_vlut32 / Q6_Vb_vlut32or pattern
 *   (each half preprocessed with Q6_Vb_vshuff_Vb) to perform the 256-entry lookup.
 *   Handle the tail (n=1000, tail=104 bytes) with a predicated store.
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n,
                      const int8_t *tableA, const int8_t *tableB)
{
    /*
     * Build combined[256] on the stack:
     *   combined[0..127]   = tableB[0..127]   (non-negative half)
     *   combined[128..255] = tableA[0..127]   (negative half)
     *
     * Align to 128 bytes so we can load it as HVX_Vector directly.
     */
    __attribute__((aligned(128))) int8_t combined[256];
    for (int k = 0; k < 128; k++) combined[k]       = tableB[k];
    for (int k = 0; k < 128; k++) combined[128 + k] = tableA[k];

    /*
     * Preprocess both 128-byte halves with vshuff for vlut32 segment layout.
     */
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(combined));         /* combined[0..127]   */
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(combined + 128));   /* combined[128..255] */

    int vlen = 128;
    int nvec = n / vlen;
    int tail = n % vlen;

    const int8_t *src = in;
    int8_t       *dst = out;

    for (int v = 0; v < nvec; v++) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        /*
         * 8-pass 256-entry LUT lookup.
         * vlut32 selects 32 entries per pass based on index[6:5] (segment field).
         * Segments 0-3 cover combined[0..127] (sTab0 = tableB side).
         * Segments 4-7 cover combined[128..255] (sTab1 = tableA side).
         */
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
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

    /* Tail: n=1000 → tail=104 bytes. Predicated store writes only valid lanes. */
    if (tail > 0) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

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
        Q6_vmem_QRIV(mask, (HVX_Vector *)dst, res);
    }
}
