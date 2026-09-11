/* HMX fp16 attention QK^T with DEEP D=128 reduction (tolerance-correct) --
 * the ACCELERATED expert. Packs Q,K into the fp16 crouton layout
 * (off(r,c)=(r/2)*64+c*2+(r&1)) one 32-deep D-tile at a time; Q6_mxclracc_hf
 * clears the persistent float accumulator ONCE, then all four
 * (activation, weight) load-matmul pairs accumulate into it before the
 * single store. K is stored [S,D] row-major (row j IS key vector j) --
 * already K^T-shaped relative to a generic matmul's B[K,N] operand -- so the
 * weight tile is packed straight from K with SWAPPED tile indices
 * (weight[r][c] = K^T[kt*T+r][c] = K[c][kt*T+r]), no separate transpose
 * buffer needed (unlike the baseline's Kt staging, and unlike the AV
 * sibling task where V is already B[K,N]-shaped).
 * Scalar VTCM accesses cost ~48 cycles each in timing mode, so the
 * pack/unpack is STAGED through cacheable buffers and moved to/from VTCM in
 * bulk 128B vector copies (the software analog of DMA).
 * Recipe: m1_driver/capability/probes/hmx_fp16_matmul.c + the 2026-06-12 spec. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *out, int n, int k_dim) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);  /* 2048B apart */
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    const int T = 32;                 /* crouton tile edge / D-tile depth */
    const int kt_count = k_dim / T;   /* 4 for k_dim=128 */

    Q6_mxclracc_hf();
    for (int kt = 0; kt < kt_count; kt++) {
        /* pack this 32-deep D-tile of Q (n x T) */
        for (int r = 0; r < n; r++)
            for (int c = 0; c < T; c++)
                aA[hvx_crouton_off(r, c)] = Q[r*k_dim + kt*T + c];
        /* weight[r][c] = K^T[kt*T+r][c] = K[c][kt*T+r]: read K directly with
         * swapped tile indices -- no transpose buffer needed */
        for (int r = 0; r < T; r++)
            for (int c = 0; c < n; c++)
                aB[hvx_crouton_off(r, c)] = K[c*k_dim + (kt*T + r)];
        { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        /* accumulates into the (uncleared) hf accumulator */
        Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
        Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    }
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) out[r*n + c] = aO[hvx_crouton_off(r, c)];
}
