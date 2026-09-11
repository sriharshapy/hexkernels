/* sol_03: HVX-vectorised Hamming distance.
 * Process 32 int32 pairs per HVX_Vector (128 bytes = 32 x 4 bytes).
 * Strategy:
 *   1. XOR: Q6_V_vxor_VV(va, vb) -> 128 XOR bytes (32 words).
 *   2. Per-halfword popcount: Q6_Vh_vpopcount_Vh -> popcount of each 16-bit lane.
 *      Each result halfword holds popcount(bits 15..0) of the input halfword.
 *   3. Sum adjacent halfword popcounts to get per-word popcount:
 *      word_pop = hi_half_pop + lo_half_pop.
 *      Use Q6_Vw_vdmpy_VhRb: Vw[w] = Vh[2w] + Vh[2w+1] (with Rb = 0x01 repeated).
 *
 * Q6_Vh_vpopcount_Vh: counts bits in each 16-bit element.
 * Q6_Vw_vdmpy_VhRb(Vh, Rt): Vw[w] = Vh[2w]*Rt.b0 + Vh[2w+1]*Rt.b1 (signed).
 *   With Rt = 0x01010101: both multipliers = 1, so Vw[w] = Vh[2w] + Vh[2w+1].
 *
 * Bit-exact: all 32 bits per word counted. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int32_t *a, const int32_t *b, int32_t *out, int n) {
    int i = 0;
    /* Process 32 pairs per iteration. */
    for (; i + 31 < n; i += 32) {
        HVX_Vector va = *(HVX_Vector const *)(a + i);
        HVX_Vector vb = *(HVX_Vector const *)(b + i);

        /* XOR: 128 bytes of XOR result */
        HVX_Vector vxor = Q6_V_vxor_VV(va, vb);

        /* Per-halfword popcount: each uint16 -> popcount(16 bits). */
        HVX_Vector vph = Q6_Vh_vpopcount_Vh(vxor);

        /* Sum adjacent halfwords to get per-word popcount (32-bit result).
         * Q6_Vw_vdmpy_VhRb: Vw[w] = Vh[2w]*b0 + Vh[2w+1]*b1.
         * With Rt = 0x01010101 (b0=b1=b2=b3=1). */
        HVX_Vector vpop32 = Q6_Vw_vdmpy_VhRb(vph, (int)0x01010101);

        *(HVX_Vector *)(out + i) = vpop32;
    }
    /* Scalar tail */
    for (; i < n; i++) {
        uint32_t v = (uint32_t)a[i] ^ (uint32_t)b[i];
        v = v - ((v >> 1) & 0x55555555u);
        v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
        v = (v + (v >> 4)) & 0x0F0F0F0Fu;
        out[i] = (int32_t)((v * 0x01010101u) >> 24);
    }
}
