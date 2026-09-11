/* i8_outer_product sol_01: HVX using Ww_vmpy_VhVh.
 * Pack b into int16, splat a[i] as int16, vmpy -> int32.
 * Careful: Ww_vmpy_VhVh produces a VectorPair where lo/hi each hold
 * every other element (even/odd), so we use Q6_Vw_vmpy_VhRh with
 * int16 scalar for correct element-wise multiply. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *C,
                      int M, int N) {
    /* Pack b[0..127] as int16 in two aligned 128-byte HVX vectors */
    int16_t blo16[64] __attribute__((aligned(128)));
    int16_t bhi16[64] __attribute__((aligned(128)));
    for (int j = 0; j < 64; j++) { blo16[j] = (int16_t)b[j]; bhi16[j] = (int16_t)b[j+64]; }
    HVX_Vector vblo = *(const HVX_Vector *)blo16;  /* b[0..63] as int16 */
    HVX_Vector vbhi = *(const HVX_Vector *)bhi16;  /* b[64..127] as int16 */

    int32_t tmplo[32] __attribute__((aligned(128)));
    int32_t tmphi[32] __attribute__((aligned(128)));
    int32_t tmplo2[32] __attribute__((aligned(128)));
    int32_t tmphi2[32] __attribute__((aligned(128)));

    for (int i = 0; i < M; i++) {
        int16_t ai = (int16_t)a[i];
        /* Ww_vmpy_VhVh(vblo, vsplat(ai)) -> 64 int32 in a VectorPair
         * but the lane ordering may be interleaved. Use element-wise: */
        HVX_Vector vai = Q6_Vh_vsplat_R((int32_t)ai);
        HVX_VectorPair plo = Q6_Ww_vmpy_VhVh(vblo, vai);
        HVX_VectorPair phi = Q6_Ww_vmpy_VhVh(vbhi, vai);

        /* Q6_Ww_vmpy_VhVh: the result VectorPair contains:
         *   lo = products of even-indexed lanes (0,2,4,...,62) -> 32 int32
         *   hi = products of odd-indexed lanes  (1,3,5,...,63) -> 32 int32
         * So we need to interleave them back. */
        *(HVX_Vector *)tmplo  = Q6_V_lo_W(plo);   /* even lanes of blo*ai */
        *(HVX_Vector *)tmphi  = Q6_V_hi_W(plo);   /* odd  lanes of blo*ai */
        *(HVX_Vector *)tmplo2 = Q6_V_lo_W(phi);   /* even lanes of bhi*ai */
        *(HVX_Vector *)tmphi2 = Q6_V_hi_W(phi);   /* odd  lanes of bhi*ai */

        int32_t *out = C + i * N;
        /* Interleave even/odd back: j=0->tmplo[0], j=1->tmphi[0], j=2->tmplo[1]... */
        for (int j = 0; j < 32; j++) { out[2*j] = tmplo[j]; out[2*j+1] = tmphi[j]; }
        for (int j = 0; j < 32; j++) { out[64+2*j] = tmplo2[j]; out[64+2*j+1] = tmphi2[j]; }
    }
}