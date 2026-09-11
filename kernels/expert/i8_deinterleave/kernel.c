/* sol_02: HVX using Q6_Vb_vdeal_Vb â€” deinterleaves even/odd bytes within one 128B vector.
 * Each 128-byte input chunk: vdeal splits it -> lo 64 bytes = even (a[]), hi 64 bytes = odd (b[]).
 * Process 256 input bytes (= 2 128B chunks) per iteration to produce 128 a[] and 128 b[] bytes:
 *   vdeal(v0): lo64=a[0..63],   hi64=b[0..63]
 *   vdeal(v1): lo64=a[64..127], hi64=b[64..127]
 *   Combine using Q6_V_lo_W / Q6_V_hi_W via vshuff to pack the halves.
 *
 * For n=500: 4 full 256-byte input passes (= 128 output pairs each) then scalar tail (n%128).
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *a, int8_t *b, int n) {
    /* Process 128 output pairs at a time (256 input bytes) */
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector v0 = *((HVX_UVector *)(in + 2 * i));       /* bytes 0..127  */
        HVX_Vector v1 = *((HVX_UVector *)(in + 2 * i + 128)); /* bytes 128..255 */
        /* vdeal: even bytes -> first half, odd bytes -> second half */
        HVX_Vector d0 = Q6_Vb_vdeal_Vb(v0);  /* lo64=a[i..i+63], hi64=b[i..i+63] */
        HVX_Vector d1 = Q6_Vb_vdeal_Vb(v1);  /* lo64=a[i+64..i+127], hi64=b[i+64..i+127] */
        /* Now pack: a_out = first 64 bytes of d0 ++ first 64 bytes of d1
         *           b_out = last  64 bytes of d0 ++ last  64 bytes of d1
         * Use Q6_W_vshuff_VVR(d1, d0, 64): with Rt=64 it should interleave 64-byte blocks */
        HVX_VectorPair pa = Q6_W_vshuff_VVR(d1, d0, 64);
        *((HVX_UVector *)(a + i)) = Q6_V_lo_W(pa);
        *((HVX_UVector *)(b + i)) = Q6_V_hi_W(pa);
    }
    /* scalar tail */
    for (; i < n; i++) {
        a[i] = in[2 * i];
        b[i] = in[2 * i + 1];
    }
}