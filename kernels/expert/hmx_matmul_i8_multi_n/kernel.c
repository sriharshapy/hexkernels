/* HMX int8 32x64 matmul over MULTIPLE N-tiles (bit-exact) — ACCELERATED expert.
 * The output is 32 rows x 64 cols = two 32-wide output tiles. The shared
 * activation tile is packed once; for each N-tile tj the weight columns
 * tj*32..tj*32+31 are packed, matmul'd (mxclracc per tile), 0x40-requant stored,
 * and un-crouton'd into the correct output columns. Crouton pack/unpack staged
 * via cacheable buffers + bulk 128B copies. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
                       int m, int ncol, int k) {
    uint8_t          *vAct  = (uint8_t  *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    int8_t           *vWgt  = (int8_t   *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);
    volatile uint8_t *vBias = (volatile uint8_t *)(uintptr_t)(HVX_VTCM_BASE + 0x2000u);
    uint16_t         *vOut  = (uint16_t *)(uintptr_t)(HVX_VTCM_BASE + 0x3000u);

    static uint8_t  aAct[2048]  HVX_ALIGN;
    static int8_t   aWgt[1024]  HVX_ALIGN;
    static uint16_t aOut[32*32] HVX_ALIGN;

    for (int i = 0; i < 2048; i++) aAct[i]  = 0;
    for (int i = 0; i < 2048; i++) vBias[i] = 0x40;
    const unsigned lim_a = 2047, lim_w = 1023, lim_o = 2047;
    const int T = 32, nt = ncol / T;   /* 2 N-tiles for ncol=64 */

    /* shared activation tile packed once */
    for (int i = 0; i < m; i++)
        for (int kk = 0; kk < k; kk++) aAct[hvx_hmx_i8_act_off(i, kk)] = A[i*k + kk];
    { HVX_Vector *s = (HVX_Vector *)aAct, *d = (HVX_Vector *)vAct;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }

    for (int tj = 0; tj < nt; tj++) {
        __asm__ volatile("mxclracc\n");
        for (int kk = 0; kk < k; kk++)
            for (int j = 0; j < T; j++)
                aWgt[hvx_hmx_i8_wgt_off(kk, j)] = B[kk*ncol + tj*T + j];
        { HVX_Vector *s = (HVX_Vector *)aWgt, *d = (HVX_Vector *)vWgt;
          for (int b = 0; b < 8; b++) d[b] = s[b]; }
        __asm__ volatile("{ activation.ub=mxmem(%0,%1)\n\t weight.b=mxmem(%2,%3) }\n"
                         :: "r"(vAct), "r"(lim_a), "r"(vWgt), "r"(lim_w) : "memory");
        __asm__ volatile("bias=mxmem(%0)\n" :: "r"(vBias) : "memory");
        __asm__ volatile("mxmem(%0,%1):after.uh=acc:2x1\n" :: "r"(vOut), "r"(lim_o) : "memory");
        __asm__ volatile("isync\n\t");
        { HVX_Vector *s = (HVX_Vector *)vOut, *d = (HVX_Vector *)aOut;
          for (int b = 0; b < 16; b++) d[b] = s[b]; }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < T; j++)
                out[i*ncol + tj*T + j] = sx12((int)aOut[hvx_hmx_i8_out_off(i, j)]);
    }
}
