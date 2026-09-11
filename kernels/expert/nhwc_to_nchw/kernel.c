/* NHWC->NCHW C=4, H=10, W=13 → 520 bytes total, 130 bytes/channel-plane
 *
 * Strategy:
 *   4-way byte deinterleave via two stages of Q6_W_vdeal_VVR(vzero, v, -1):
 *     Stage 1: separate c0,c2 (even positions) from c1,c3 (odd positions)
 *     Stage 2: separate c0 from c2 (and c1 from c3)
 *     Per 32-pixel input vector → 4 vectors each with 32 channel bytes in [0..31]
 *
 *   Merge 4×32-byte chunks per channel into one 128B plane vector:
 *     vror(chunk_k, (3-k)*32) positions chunk_k at bytes [k*32..(k+1)*32-1]
 *     OR the 4 shifted chunks together
 *
 *   Stores:
 *     plane 0 (out+0): 128B-aligned → *(HVX_Vector *)
 *     planes 1,2,3 (out+130, +260, +390): unaligned → *(HVX_UVector *)
 *     2-byte scalar tail for pixels 128..129 per channel
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

void candidate_kernel(const int8_t *in, int8_t *out, int C, int H, int W) {
    /* Pinned: C=4, H=10, W=13 → PLANE=130, TOTAL=520 */
    const int PLANE = H * W;  /* 130 */

    HVX_Vector vzero = Q6_V_vzero();

    /* Load 4 full 128B vectors (32 pixels × 4ch each) */
    HVX_Vector v0 = *(const HVX_Vector *)(in + 0);
    HVX_Vector v1 = *(const HVX_Vector *)(in + 128);
    HVX_Vector v2 = *(const HVX_Vector *)(in + 256);
    HVX_Vector v3 = *(const HVX_Vector *)(in + 384);

/* Macro: 2-stage vdeal to extract 32 bytes of each channel from a 32-pixel input vector.
 * After deal, each cX vector has: channel values in bytes [0..31], zeros in [32..127]. */
#define DEINT4(vin, oc0, oc1, oc2, oc3)                          \
    do {                                                           \
        HVX_VectorPair _s1 = Q6_W_vdeal_VVR(vzero, (vin), -1);   \
        HVX_Vector _lo1 = Q6_V_lo_W(_s1); /* [c0,c2, c0,c2,...] 64B valid */ \
        HVX_Vector _hi1 = Q6_V_hi_W(_s1); /* [c1,c3, c1,c3,...] 64B valid */ \
        HVX_VectorPair _s2a = Q6_W_vdeal_VVR(vzero, _lo1, -1);   \
        HVX_VectorPair _s2b = Q6_W_vdeal_VVR(vzero, _hi1, -1);   \
        (oc0) = Q6_V_lo_W(_s2a); /* c0 in [0..31] */ \
        (oc2) = Q6_V_hi_W(_s2a); /* c2 in [0..31] */ \
        (oc1) = Q6_V_lo_W(_s2b); /* c1 in [0..31] */ \
        (oc3) = Q6_V_hi_W(_s2b); /* c3 in [0..31] */ \
    } while (0)

    HVX_Vector c0_0, c1_0, c2_0, c3_0;
    HVX_Vector c0_1, c1_1, c2_1, c3_1;
    HVX_Vector c0_2, c1_2, c2_2, c3_2;
    HVX_Vector c0_3, c1_3, c2_3, c3_3;

    DEINT4(v0, c0_0, c1_0, c2_0, c3_0);
    DEINT4(v1, c0_1, c1_1, c2_1, c3_1);
    DEINT4(v2, c0_2, c1_2, c2_2, c3_2);
    DEINT4(v3, c0_3, c1_3, c2_3, c3_3);

#undef DEINT4

    /* Merge 4×32-byte chunks into one 128B plane vector.
     * vror(v, R): result[k] = v[(k+R) mod 128].
     * To place chunk_k at bytes [k*32 .. k*32+31], use vror(chunk_k, (3-k)*32):
     *   chunk_0: vror(v, 96): result[k]=v[(k+96)%128]; at k=0..31 → v[96..127]=0, WRONG!
     *
     * Wait: vror right-rotates. To move data from [0..31] to [32..63]:
     *   I need result[32..63] = src[0..31]
     *   result[k] = src[(k+R)%128]  →  result[32] = src[(32+R)%128] = src[0] → R=96? No:
     *   (32+R)%128 = 0 → R = 96. Then result[32] = src[96], not src[0]. WRONG.
     *
     * I got confused. Let me use valign instead.
     * Q6_V_valign_VVR(Vu, Vv, R): extract 128B starting at byte R from Vv:Vu (Vv=hi, Vu=lo).
     * To shift chunk from [0..31] to [32..63] in output: I want output[32+k] = chunk[k].
     * Use valign(chunk, vzero, R) = bytes R..R+127 of vzero:chunk = [chunk[R..127], vzero[0..R-1]].
     * To get chunk[0..31] at positions 32..63: need output[32..63]=chunk[0..31].
     * This means I want the output vector to have chunk at an offset of 32.
     * Use vror LEFT rotation equivalent: left-shift by 32 = right-rotate by 128-32=96.
     * vror(v, 96): result[k] = v[(k+96)%128].
     *   k=32 → v[32+96=128→0] = v[0] = c0(p32). WAIT: (32+96)%128 = 128%128 = 0. ✓
     *   k=63 → v[(63+96)%128] = v[159%128] = v[31]. ✓
     *   k=0  → v[96] = 0. ✓ (zeros are in positions 32..127 of chunk vectors)
     * So vror(chunk, 96) puts chunk data at output positions 32..63. ✓ for chunk_1.
     *
     * Summary:
     *   chunk_0 at [0..31]:   vror(chunk_0,  0) = chunk_0 (no rotation)
     *   chunk_1 at [32..63]:  vror(chunk_1, 96)
     *   chunk_2 at [64..95]:  vror(chunk_2, 64)
     *   chunk_3 at [96..127]: vror(chunk_3, 32)
     */
#define MERGE4(c0, c1, c2, c3)  \
    Q6_V_vor_VV(Q6_V_vor_VV((c0), Q6_V_vror_VR((c1), 96)), \
                Q6_V_vor_VV(Q6_V_vror_VR((c2), 64), Q6_V_vror_VR((c3), 32)))

    HVX_Vector plane0 = MERGE4(c0_0, c0_1, c0_2, c0_3);
    HVX_Vector plane1 = MERGE4(c1_0, c1_1, c1_2, c1_3);
    HVX_Vector plane2 = MERGE4(c2_0, c2_1, c2_2, c2_3);
    HVX_Vector plane3 = MERGE4(c3_0, c3_1, c3_2, c3_3);

#undef MERGE4

    /* Store 128 bytes of each channel plane (covers pixels 0..127).
     * Plane 0 is 128B-aligned. Planes 1,2,3 are misaligned → use HVX_UVector. */
    *(HVX_Vector  *)(out + 0 * PLANE) = plane0;
    *(HVX_UVector *)(out + 1 * PLANE) = plane1;
    *(HVX_UVector *)(out + 2 * PLANE) = plane2;
    *(HVX_UVector *)(out + 3 * PLANE) = plane3;

    /* Scalar tail: pixels 128..129 (2 pixels × 4 channels = 8 bytes) */
    for (int p = 128; p < PLANE; p++) {
        out[0*PLANE + p] = in[p*4 + 0];
        out[1*PLANE + p] = in[p*4 + 1];
        out[2*PLANE + p] = in[p*4 + 2];
        out[3*PLANE + p] = in[p*4 + 3];
    }
}
