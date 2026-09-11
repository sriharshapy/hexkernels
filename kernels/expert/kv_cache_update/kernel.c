/* kv_cache_update HVX v2:
 * Load k_new/v_new as one 128B HVX vector each.
 * For each head h, rotate the vector to place head h's 32 bytes at the
 * correct 128B-block position, then use a predicated vmem store (Q6_vmem_QRIV)
 * to write exactly those 32 bytes without touching adjacent cache slots.
 *
 * Pinned: num_heads=4, head_dim=32, cache_len=64.
 * dest offset = (h*cache_len + pos)*head_dim.
 * The 128B-aligned base of that block is at offset & ~127.
 * byte_off = offset & 127 is always a multiple of 32 (since head_dim=32).
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(int8_t *k_cache, int8_t *v_cache,
                      const int8_t *k_new, const int8_t *v_new,
                      int pos, int num_heads, int cache_len, int head_dim) {
    /* Load full 128B new vectors (k_new, v_new are 128B-aligned, 128B each). */
    HVX_Vector vk = *(const HVX_Vector *)k_new;
    HVX_Vector vv = *(const HVX_Vector *)v_new;

    int hd = head_dim; /* 32 */

    for (int h = 0; h < num_heads; h++) {
        int offset   = (h * cache_len + pos) * hd;     /* byte offset from cache base */
        int blk_base = offset & ~127;                   /* 128B-aligned block start */
        int byte_off = offset & 127;                    /* 0, 32, 64, or 96 */

        /* Rotate so head h's 32 bytes land at byte_off within the 128B block.
         * Q6_V_vror_VR(v, r): byte that was at position i is now at (i - r) mod 128.
         * We want byte h*hd to end up at byte_off: (h*hd - r) mod 128 = byte_off
         *   => r = (h*hd - byte_off + 128) % 128 */
        int rot = ((h * hd) - byte_off + 128) & 127;
        HVX_Vector rk = Q6_V_vror_VR(vk, rot);
        HVX_Vector rv = Q6_V_vror_VR(vv, rot);

        /* Predicate: select bytes [byte_off, byte_off+32). */
        HVX_VectorPred qk, qv;
        if (byte_off == 0) {
            qk = Q6_Q_vsetq2_R(32);
            qv = Q6_Q_vsetq2_R(32);
        } else if (byte_off == 96) {
            qk = Q6_Q_not_Q(Q6_Q_vsetq2_R(96));
            qv = Q6_Q_not_Q(Q6_Q_vsetq2_R(96));
        } else {
            qk = Q6_Q_xor_QQ(Q6_Q_vsetq2_R(byte_off + 32), Q6_Q_vsetq2_R(byte_off));
            qv = Q6_Q_xor_QQ(Q6_Q_vsetq2_R(byte_off + 32), Q6_Q_vsetq2_R(byte_off));
        }

        /* Predicated store: only write the 32 bytes we own. */
        Q6_vmem_QRIV(qk, (HVX_Vector *)(k_cache + blk_base), rk);
        Q6_vmem_QRIV(qv, (HVX_Vector *)(v_cache + blk_base), rv);
    }
}
