/* Near-miss: correct row-chained group DMA, but H is not handled generically
 * -- it silently drops the last partial row-group (H=136 is not a multiple
 * of RPC=16, so the last 8 rows of out[] are never written and stay
 * poisoned at 0xA5). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define RPC 16

static desc_t d_in[RPC];
static desc_t d_out;

void candidate_kernel(const int8_t *a, int8_t *out, int H, int W, int rowstride) {
    const uint32_t vt0 = VTCM_BASE;
    int ngroups = H / RPC;   /* BUG: remainder rows silently dropped below */

    for (int g = 0; g < ngroups; g++) {
        for (int r = 0; r < RPC; r++) {
            d_in[r].next = (r + 1 < RPC) ? (uint32_t)(uintptr_t)&d_in[r + 1] : 0;
            d_in[r].ctrl = (uint32_t)W;
            d_in[r].src  = (uint32_t)(uintptr_t)(a + (long)(g * RPC + r) * rowstride);
            d_in[r].dst  = vt0 + (uint32_t)r * (uint32_t)W;
        }
        Q6_dmstart_A(&d_in[0]); Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = (uint32_t)(RPC * W);
        d_out.src = vt0; d_out.dst = (uint32_t)(uintptr_t)(out + (long)g * RPC * W);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
    /* No handling of the trailing H % RPC rows. */
}
