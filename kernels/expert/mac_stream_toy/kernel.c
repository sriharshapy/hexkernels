/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Operand-streaming block-MAC pipeline: DMA the a and b tile for block m+1 into
 * the other VTCM slot (async) while the vrmpy accumulator chews on block m in
 * the current slot; then horizontally reduce and emit out[m]. Double-buffering
 * hides DDR latency; the MAC reads its operands from fast on-chip VTCM.
 * One block == one CH-byte tile.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile == block size */

static desc_t d_a, d_b, d_pa, d_pb;

void candidate_kernel(const uint8_t *a, const int8_t *b, int32_t *out, int n, int blk) {
    /* This expert is specialized to blk == CH (the toy's fixed block). */
    int nb = n / CH;
    const uint32_t va0 = VTCM_BASE,        va1 = VTCM_BASE + CH;
    const uint32_t vb0 = VTCM_BASE + 2*CH, vb1 = VTCM_BASE + 3*CH;
    int vpb = CH / 128;

    if (nb == 0) return;

    /* Prologue: DMA block 0 (a,b) into slot 0 via a chained descriptor. */
    d_a.next = (uint32_t)(uintptr_t)&d_b; d_a.ctrl = CH; d_a.src = (uint32_t)(uintptr_t)a; d_a.dst = va0;
    d_b.next = 0; d_b.ctrl = CH; d_b.src = (uint32_t)(uintptr_t)b; d_b.dst = vb0;
    Q6_dmstart_A(&d_a); Q6_R_dmwait();

    for (int m = 0; m < nb; m++) {
        uint32_t ca = (m & 1) ? va1 : va0, cb = (m & 1) ? vb1 : vb0;
        uint32_t na = (m & 1) ? va0 : va1, nb_ = (m & 1) ? vb0 : vb1;

        if (m + 1 < nb) {   /* async prefetch next block */
            d_pa.next = (uint32_t)(uintptr_t)&d_pb; d_pa.ctrl = CH;
            d_pa.src = (uint32_t)(uintptr_t)(a + (m+1)*CH); d_pa.dst = na;
            d_pb.next = 0; d_pb.ctrl = CH;
            d_pb.src = (uint32_t)(uintptr_t)(b + (m+1)*CH); d_pb.dst = nb_;
            Q6_dmstart_A(&d_pa);
        }

        const HVX_Vector *pa = (const HVX_Vector *)(uintptr_t)ca;
        const HVX_Vector *pb = (const HVX_Vector *)(uintptr_t)cb;
        HVX_Vector acc = Q6_V_vzero();
        for (int v = 0; v < vpb; v++)
            acc = Q6_Vw_vrmpyacc_VwVubVb(acc, pa[v], pb[v]);

        if (m + 1 < nb) Q6_R_dmwait();

        int32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = acc;
        int32_t s = 0;
        for (int j = 0; j < 32; j++) s += lanes[j];
        out[m] = s;
    }
}
